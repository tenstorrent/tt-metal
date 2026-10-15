#!/usr/bin/env python3
"""Energy per FLOP: naive vs blocked access pattern, both on Blackhole p100a.

Same chart format as compare_runs2.py. Both series are POWER_CASE=0 on the same
board with the same shape and iteration count; the only difference is whether the
kernel reuses tiles.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOPS = 2 * 1024 * 2048 * 2048 * 160
SHAPE = "M=1024 N=2048 K=2048 x160 iterations, HiFi4 bfloat16, split mode, POWER_CASE=0"
LABELS = (
    "Naive, no tile reuse (2.00 tile reads per multiply)",
    "Blocked 2x4, tile reuse (0.75 tile reads per multiply)",
)
GRIDS = ["3x3", "3x4", "4x3", "4x4", "4x5", "5x4", "5x5", "5x6", "6x5",
         "6x6", "6x7", "7x6", "7x7", "8x7", "9x8", "10x9", "11x10"]


def pj_per_flop(path):
    d = {r["grid"]: r for r in csv.DictReader(open(path))}
    return {g: float(d[g]["energy_j_vi"]) / FLOPS * 1e12 for g in GRIDS if g in d}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("naive_csv", type=Path, help="program_intervals.csv of the naive POWER_CASE=0 run")
    ap.add_argument("blocked_csv", type=Path, help="program_intervals.csv of the HIGH_POWER_BLOCK_M=2 N=4 run")
    ap.add_argument("--out", type=Path, default=Path("pj_per_flop_blackhole_naive_vs_blocked.png"))
    args = ap.parse_args()

    series = [(label, pj_per_flop(path)) for path, label in zip((args.naive_csv, args.blocked_csv), LABELS)]
    x = np.arange(len(GRIDS))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(max(10, len(GRIDS) * 1.0), 6))
    for i, (label, vals) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, [vals.get(g, float("nan")) for g in GRIDS], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(GRIDS, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel("Energy per FLOP [pJ]")
    ax.set_title(f"Energy per FLOP per grid - Blackhole p100a, naive vs blocked\n{SHAPE}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = args.out
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
