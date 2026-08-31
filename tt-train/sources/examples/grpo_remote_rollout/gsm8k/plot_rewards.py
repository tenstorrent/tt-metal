#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Plot ``grpo_metrics.csv`` produced by ``gsm8k_training_example.py``.

Three panels:
  1. Total reward per step  (raw + EMA-N)
  2. Reward components      (correctness / xmlcount / soft_format / strict_format / int_reward)
  3. Average completion length

Usage:
    python plot_rewards.py generated/tt-train/grpo_gsm8k_run/<utc>/grpo_metrics.csv
    python plot_rewards.py <csv> --out rewards.png --ema 10 --title "my run"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COMPONENT_COLS = (
    "correctness_reward_mean",
    "xmlcount_reward_mean",
    "soft_format_reward_mean",
    "strict_format_reward_mean",
    "int_reward_mean",
)
COMPONENT_LABELS = {
    "correctness_reward_mean": "correctness",
    "xmlcount_reward_mean": "xmlcount",
    "soft_format_reward_mean": "soft_format",
    "strict_format_reward_mean": "strict_format",
    "int_reward_mean": "int_reward",
}
COMPONENT_MAX = {
    "correctness_reward_mean": 2.0,
    "xmlcount_reward_mean": 0.5,
    "soft_format_reward_mean": 0.5,
    "strict_format_reward_mean": 0.5,
    "int_reward_mean": 0.5,
}
COMPONENT_COLORS = {
    "correctness_reward_mean": "#1f6feb",
    "xmlcount_reward_mean": "#f97316",
    "soft_format_reward_mean": "#16a34a",
    "strict_format_reward_mean": "#e11d48",
    "int_reward_mean": "#eab308",
}


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", type=Path, help="Path to grpo_metrics.csv.")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <csv_dir>/grpo_metrics.png).")
    p.add_argument("--ema", type=int, default=10, help="EMA span for the total-reward smoother.")
    p.add_argument("--title", type=str, default="GRPO on Qwen3-0.6B-Base-Think-SFT / GSM8K")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    out_path = args.out or (args.csv.parent / "grpo_metrics.png")

    required = ("step", "reward_mean", "mean_completion_len", *COMPONENT_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing expected columns: {missing}.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(args.title, fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(df["step"], df["reward_mean"], color="#9ec5ff", linewidth=1.0, label="raw")
    ax.plot(df["step"], ema(df["reward_mean"], args.ema), color="#1f6feb", linewidth=2.0, label=f"EMA-{args.ema}")
    peak = df["reward_mean"].ewm(span=args.ema, adjust=False).mean().max()
    ax.axhline(peak, color="#1f6feb", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.annotate(f"peak {peak:.2f}", xy=(df["step"].iloc[-1], peak), fontsize=9, color="#1f6feb")
    ax.set_title("Total reward per step  (raw + EMA-{})".format(args.ema))
    ax.set_xlabel("step")
    ax.set_ylabel("reward (max 4.0)")
    ax.set_ylim(0, 4.2)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    for c in COMPONENT_COLS:
        label = f"{COMPONENT_LABELS[c]} (max {COMPONENT_MAX[c]})"
        ax.plot(df["step"], df[c], color=COMPONENT_COLORS[c], linewidth=1.4, label=label)
    ax.set_title("Reward components")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(df["step"], df["mean_completion_len"], color="#0891b2", linewidth=1.0, label="raw")
    ax.plot(
        df["step"], ema(df["mean_completion_len"], args.ema), color="#0e7490", linewidth=2.0, label=f"EMA-{args.ema}"
    )
    ax.set_title("Average completion length")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
