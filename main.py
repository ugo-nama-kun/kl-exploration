import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Parameters
n_action = 20  # Even number
n_observation = n_action  # Even number
eps = 0.1  # prior count (small: white paper, large: stick to the prior)
temperature = 1
n_trial = 300

# Environment
def env_prob(action, is_deterministic=False):
    # o = a
    if is_deterministic:
        obs = action
        return obs

    # stochastic
    if action in (0, n_action-1):
        if action == 0:
            obs = np.random.choice([0, 1])
        else:
            obs = np.random.choice([n_action-2, n_action-1])
    else:
        obs = np.random.choice([action-1, action, action+1])
    return obs


# Agent
weight = np.array([i for i in range(int(n_observation/2))] + [i for i in range(int(n_observation/2), 0, -1)]) + 0.1
np.random.shuffle(weight)
prob_star = weight/np.sum(weight)
print(prob_star)

model_count = np.zeros((n_action, n_observation), dtype=np.float) + eps


def prob_model(obs, action):
    p_o = model_count[action]/np.sum(model_count[action])
    return p_o[obs]

def take_action():
    q_val = np.zeros(n_action)
    for a in range(n_action):
        for o in range(n_observation):
            q_val[a] += prob_model(o, a) * (-np.log(prob_star[o]) + temperature * np.log(prob_model(o, a)))
    best_action = np.argmin(q_val)
    return best_action

# Main Experiment


def main():
    visit_counts = np.zeros(n_observation)
    returns = [None] * n_trial
    fig, ax = plt.subplots(3, 1)
    plt.pause(0.001)
    for n in tqdm(range(n_trial)):
        action = take_action()

        # interaction
        obs = env_prob(action)

        # model update
        model_count[action, obs] += 1

        # visualization
        visit_counts[obs] += 1
        returns[n] = - np.log(prob_star[obs])

        for p_ in ax:
            p_.cla()
        ax[0].bar(x=range(n_observation), height=prob_star, color='b')
        ax[1].bar(x=range(n_observation), height=visit_counts, color='b')
        ax[2].plot(range(n_trial), returns, color='r')
        ax[1].plot([action], [20], '*k')
        plt.pause(0.01)


if __name__ == '__main__':
    main()
    plt.show()
    print("done")
