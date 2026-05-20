# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate the four charts for the TEN-4679 Confluence page.

Reads the post-processed CSV (one row per seq_len) and produces:
  1. Stacked bar of cycle breakdown (FPU / SFPU / idle) vs S
  2. Line plot of FPU% and SFPU% vs S
  3. FPU/SFPU ratio across S
  4. Single-cycle-exp counterfactual (today vs hypothetical)

Usage:
    python analysis/make_charts.py analysis/ten4679_results.csv -o analysis/charts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def chart_stacked_breakdown(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df["seq_len"].astype(str)
    fpu = df["fpu_pct"]
    sfpu = df["sfpu_pct"]
    idle = df["math_idle_residual_pct"]

    ax.bar(x, fpu, label="FPU active", color="#1f77b4")
    ax.bar(x, sfpu, bottom=fpu, label="SFPU active", color="#ff7f0e")
    ax.bar(x, idle, bottom=fpu + sfpu, label="Math-thread idle (incl. implicit dest stall)", color="#d62728", alpha=0.7)

    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("% of math-thread cycles")
    ax.set_title(
        "Single-chip SDPA on Black Hole — math-thread cycle breakdown\n"
        "Llama 3.1 8B prefill: nh=32, nkv=8, dh=128, BF8, causal"
    )
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_stacked_breakdown.png", dpi=140)
    plt.close(fig)


def chart_fpu_sfpu_lines(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["seq_len"], df["fpu_pct"], marker="o", linewidth=2, label="FPU active %", color="#1f77b4")
    ax.plot(df["seq_len"], df["sfpu_pct"], marker="s", linewidth=2, label="SFPU active %", color="#ff7f0e")
    ax.plot(
        df["seq_len"],
        df["math_pct"],
        marker="^",
        linewidth=2,
        label="MATH active %  (= FPU + SFPU)",
        color="#2ca02c",
        linestyle="--",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(df["seq_len"])
    ax.set_xticklabels([str(s) for s in df["seq_len"]])
    ax.set_xlabel("Sequence length (tokens, log scale)")
    ax.set_ylabel("% of math-thread cycles")
    ax.set_title("FPU% and SFPU% vs sequence length")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02_fpu_sfpu_vs_S.png", dpi=140)
    plt.close(fig)


def chart_ratio(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ratio = df["fpu_pct"] / df["sfpu_pct"]
    ax.plot(df["seq_len"], ratio, marker="o", linewidth=2, color="#9467bd")
    ax.axhline(2.0, linestyle="--", color="grey", alpha=0.5, label="reference: 2:1")
    ax.set_xscale("log", base=2)
    ax.set_xticks(df["seq_len"])
    ax.set_xticklabels([str(s) for s in df["seq_len"]])
    ax.set_xlabel("Sequence length (tokens, log scale)")
    ax.set_ylabel("FPU% / SFPU%")
    ax.set_ylim(0, 3)
    ax.set_title("FPU/SFPU ratio across sequence lengths\n(structural: ≈ 2·DH / (c_exp + c_reduce))")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_fpu_sfpu_ratio.png", dpi=140)
    plt.close(fig)


def chart_counterfactual(df: pd.DataFrame, out: Path, exp_share: float = 0.5) -> None:
    """Single-cycle-exp counterfactual: assume `exp` is `exp_share` of SFPU, drops to ~0."""
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(df))

    today_fpu = df["fpu_pct"].values
    today_sfpu = df["sfpu_pct"].values
    cf_sfpu = today_sfpu * (1 - exp_share)

    ax.bar(x - width / 2, today_fpu, width, label="FPU active (today)", color="#1f77b4")
    ax.bar(x - width / 2, today_sfpu, width, bottom=today_fpu, label="SFPU active (today)", color="#ff7f0e")

    ax.bar(x + width / 2, today_fpu, width, label="FPU active (1-cycle exp)", color="#1f77b4", hatch="///", alpha=0.85)
    ax.bar(
        x + width / 2,
        cf_sfpu,
        width,
        bottom=today_fpu,
        label="SFPU active (1-cycle exp)",
        color="#ff7f0e",
        hatch="///",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in df["seq_len"]])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("% of math-thread cycles")
    ax.set_title(f"Single-cycle exp counterfactual\n(assumes exp ≈ {int(exp_share*100)}% of SFPU work; FPU unchanged)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    for i in range(len(df)):
        drop = today_sfpu[i] - cf_sfpu[i]
        ax.text(
            i + width / 2, today_fpu[i] + cf_sfpu[i] + 1.5, f"−{drop:.1f} pp", ha="center", fontsize=8, color="#555555"
        )

    fig.tight_layout()
    fig.savefig(out / "04_counterfactual_single_cycle_exp.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("analysis/charts"))
    ap.add_argument(
        "--exp-share", type=float, default=0.5, help="Fraction of SFPU work attributable to exp (default 0.5)"
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv).sort_values("seq_len").reset_index(drop=True)
    if "counter_overflow" in df.columns:
        before = len(df)
        df = df[df["counter_overflow"] == "no"].reset_index(drop=True)
        if len(df) < before:
            print(f"Note: dropped {before - len(df)} rows with counter overflow.")

    chart_stacked_breakdown(df, args.outdir)
    chart_fpu_sfpu_lines(df, args.outdir)
    chart_ratio(df, args.outdir)
    chart_counterfactual(df, args.outdir, args.exp_share)
    print(f"Wrote 4 charts to {args.outdir}")


if __name__ == "__main__":
    main()
