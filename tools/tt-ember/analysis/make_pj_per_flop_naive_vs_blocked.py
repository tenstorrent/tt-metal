#!/usr/bin/env python3
"""Energy per FLOP: naive vs blocked access pattern (paper Fig. 2).

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

LABELS = (
    "Naive, no tile reuse (2.00 tile reads per multiply)",
    "Blocked 2x4, tile reuse (0.75 tile reads per multiply)",
)


def load(path):
    return {r["grid"]: r for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("naive_csv", type=Path, help="program_intervals.csv of the naive POWER_CASE=0 run")
    ap.add_argument("blocked_csv", type=Path, help="program_intervals.csv of the LONG_MATMUL_BLOCK_M=2 N=4 run")
    ap.add_argument(
        "--app-args",
        nargs=4,
        type=int,
        required=True,
        metavar=("M", "N", "K", "ITERS"),
        help="Workload shape and iteration count both runs used",
    )
    ap.add_argument("--board", default="", help="Board name for the chart title, e.g. 'Blackhole p100a'")
    ap.add_argument(
        "--keep-first-grid",
        action="store_true",
        help="Include the first interval; its baseline is one-sided and unreliable.",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("pj_per_flop_naive_vs_blocked.png"))
    args = ap.parse_args()

    m, n, k, iters = args.app_args
    flops = 2 * m * n * k * iters
    shape = f"M={m} N={n} K={k} x{iters} iterations, HiFi4 bfloat16, split mode, POWER_CASE=0"

    runs = [load(args.naive_csv), load(args.blocked_csv)]
    # Grid order comes from the naive run, so the chart follows whatever the device swept.
    grids = [g for g in runs[0] if g in runs[1]]
    if not args.keep_first_grid:
        grids = grids[1:]
    series = [
        (label, {g: float(run[g]["energy_j_vi"]) / flops * 1e12 for g in grids}) for run, label in zip(runs, LABELS)
    ]
    x = np.arange(len(grids))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(max(10, len(grids) * 1.0), 6))
    for i, (label, vals) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, [vals[g] for g in grids], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel("Energy per FLOP [pJ]")
    board = f" - {args.board}" if args.board else ""
    ax.set_title(f"Energy per FLOP per grid{board}, naive vs blocked\n{shape}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
