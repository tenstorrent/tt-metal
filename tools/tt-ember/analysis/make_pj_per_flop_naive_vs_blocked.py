#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Energy per FLOP: naive vs blocked access pattern (paper Fig. 2).

Same chart format as compare_cases.py. Both series are the same POWER_CASE (0 in the paper)
on the same board with the same shape and iteration count; the only difference is whether the
kernel reuses tiles. --block and --case label the chart; run_sweep.py passes the ones it ran.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# common.py lives one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import PJ_PER_J, add_mode_argument, save_figure, whole_run_value  # noqa: E402


def reads_per_multiply(bm: int, bn: int) -> float:
    """DRAM tile reads per tile multiply with BM x BN output blocking (2.00 at 1x1)."""
    return (bm + bn) / (bm * bn)


def load(path):
    return {r["grid"]: r for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("naive_csv", type=Path, help="program_intervals.csv of the naive POWER_CASE=0 run")
    ap.add_argument("blocked_csv", type=Path, help="program_intervals.csv of the blocked run")
    ap.add_argument(
        "--block", nargs=2, type=int, default=(2, 4), metavar=("M", "N"), help="Block of the blocked run. Default: 2 4"
    )
    ap.add_argument("--case", default="regular", help="Case subdir both runs used, for the title. Default: regular")
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
    add_mode_argument(ap)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("pj_per_flop_naive_vs_blocked.png"))
    args = ap.parse_args()

    m, n, k, iters = args.app_args
    flops = 2 * m * n * k * iters
    shape = f"M={m} N={n} K={k} x{iters} iterations, HiFi4 bfloat16, split mode, case {args.case}"
    bm, bn = args.block
    labels = (
        f"Naive, no tile reuse ({reads_per_multiply(1, 1):.2f} tile reads per multiply)",
        f"Blocked {bm}x{bn}, tile reuse ({reads_per_multiply(bm, bn):.2f} tile reads per multiply)",
    )

    runs = [load(args.naive_csv), load(args.blocked_csv)]
    # Grid order comes from the naive run, so the chart follows whatever the device swept.
    grids = [g for g in runs[0] if g in runs[1]]
    if not args.keep_first_grid:
        grids = grids[1:]
    series = [
        (label, {g: whole_run_value(run[g], "energy_j_vi", args.mode == "legacy") / flops * PJ_PER_J for g in grids})
        for run, label in zip(runs, labels)
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
    save_figure(fig, args.out, args.dpi)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
