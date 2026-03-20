#!/usr/bin/env python3
import re
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(log_path):
    pattern = re.compile(r"reward_mean=([\d.]+),\s*reward_std=([\d.]+)")

    means = []
    stds = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                means.append(float(m.group(1)))
                stds.append(float(m.group(2)))

    if not means:
        print("No reward_mean entries found in log.")
        sys.exit(1)

    steps = np.arange(1, len(means) + 1)
    means = np.array(means)
    stds = np.array(stds)

    window = min(20, len(means) // 4) or 1
    mean_smooth = np.convolve(means, np.ones(window) / window, mode="valid")
    std_smooth = np.convolve(stds, np.ones(window) / window, mode="valid")
    steps_smooth = steps[window - 1 :]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.scatter(steps, means, alpha=0.15, s=8, color="steelblue")
    ax1.plot(steps_smooth, mean_smooth, color="steelblue", linewidth=2, label=f"rolling avg (w={window})")
    ax1.set_ylabel("reward_mean")
    ax1.set_title("GRPO Training Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(steps, stds, alpha=0.15, s=8, color="coral")
    ax2.plot(steps_smooth, std_smooth, color="coral", linewidth=2, label=f"rolling avg (w={window})")
    ax2.set_ylabel("reward_std")
    ax2.set_xlabel("step")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.splitext(log_path)[0] + "_rewards.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_grpo_training.log>")
        sys.exit(1)
    plot_rewards(sys.argv[1])
