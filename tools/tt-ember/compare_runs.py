#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare power/performance metrics across multiple measured runs.

Usage:
    # All runs in out/
    python3 compare_runs.py --out-root out/ --output-dir out/comparison

    # Only runs starting with 'prefill'
    python3 compare_runs.py --out-root out/ --filter prefill --output-dir out/comparison

    # Specific runs
    python3 compare_runs.py --out-root out/ --filter prefill_ fixed_tpc_
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DEFAULT_DPI, save_figure

# Every chart here overlays all runs on one wide axis.
FIGSIZE = (14, 6)

METRICS = {
    "duration": ("window_s", "Duration [s]", "Duration vs Grid Size"),
    "charge_total_avg": ("charge_avg_mah", "Charge total avg [mAh]", "Total Charge (Avg Current) vs Grid Size"),
    "charge_total_peak": ("charge_peak_mah", "Charge total peak [mAh]", "Total Charge (Peak Current) vs Grid Size"),
    "charge_dynamic_avg": (
        "dynamic_charge_avg_mah",
        "Dynamic charge avg [mAh]",
        "Dynamic Charge (Avg Current) vs Grid Size",
    ),
    "charge_dynamic_peak": (
        "dynamic_charge_peak_mah",
        "Dynamic charge peak [mAh]",
        "Dynamic Charge (Peak Current) vs Grid Size",
    ),
    "current_total_avg": ("avg_current_a", "Avg current [A]", "Total Average Current vs Grid Size"),
    "current_total_peak": ("peak_current_a", "Peak current [A]", "Total Peak Current vs Grid Size"),
    "current_dynamic_avg": ("dynamic_avg_current_a", "Dynamic avg current [A]", "Dynamic Average Current vs Grid Size"),
    "current_dynamic_peak": ("dynamic_peak_current_a", "Dynamic peak current [A]", "Dynamic Peak Current vs Grid Size"),
}


def load_runs(out_root: Path, filters: list[str]) -> dict[str, pd.DataFrame]:
    """Return {run_name: dataframe} for each matching subdirectory."""
    runs = {}
    for subdir in sorted(out_root.iterdir()):
        if not subdir.is_dir():
            continue
        if filters and not any(subdir.name.startswith(f) for f in filters):
            continue
        csv = subdir / "program_intervals.csv"
        if not csv.is_file():
            print(f"  WARNING: no program_intervals.csv in {subdir.name}, skipping.")
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        runs[subdir.name] = df
    return runs


def grid_union(runs: dict[str, pd.DataFrame]) -> list[str]:
    """Every grid that appears in any run, ordered by core count (then by name)."""
    grids = {str(g) for df in runs.values() for g in df["grid"]}

    def cores(grid: str) -> int:
        x, _, y = grid.partition("x")
        return int(x) * int(y) if x.isdigit() and y.isdigit() else 0

    return sorted(grids, key=lambda g: (cores(g), g))


def x_positions(df: pd.DataFrame, all_grids: list[str]) -> list[int]:
    """Each row's position on the shared grid axis."""
    return [all_grids.index(str(g)) for g in df["grid"]]


def finish_and_save(fig, ax, all_grids: list[str], ylabel: str, title: str, out_path: Path, dpi: int) -> None:
    ax.set_xticks(range(len(all_grids)))
    ax.set_xticklabels(all_grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")

    save_figure(fig, out_path, dpi)
    print(f"  Saved: {out_path}")


def plot_metric(runs: dict[str, pd.DataFrame], column: str, ylabel: str, title: str, out_path: Path, dpi: int) -> None:
    """One PNG: all runs overlaid on the same axes, x = grid string."""

    all_grids = grid_union(runs)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for run_name, df in runs.items():
        ax.plot(x_positions(df, all_grids), df[column].tolist(), marker="o", label=run_name)

    finish_and_save(fig, ax, all_grids, ylabel, title, out_path, dpi)


def plot_overcompute(runs: dict[str, pd.DataFrame], column: str, out_path: Path, dpi: int) -> None:
    """Plot (ideal - measured) for each run on one graph."""

    all_grids = grid_union(runs)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for run_name, df in runs.items():
        if column not in df.columns:
            continue

        cores = df["cores"].to_numpy(dtype=float)
        values = df[column].to_numpy(dtype=float)

        finite = np.where(np.isfinite(values))[0]
        if finite.size == 0:
            continue

        i0 = int(finite[0])
        c0, y0 = cores[i0], values[i0]
        if c0 == 0 or not np.isfinite(y0):
            continue

        ideal = y0 * (cores / c0)
        overcompute = ideal - values

        ax.plot(x_positions(df, all_grids), overcompute, marker="o", label=run_name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    finish_and_save(
        fig,
        ax,
        all_grids,
        "Ideal − Measured [mAh]",
        "Overcompute (gap between ideal and measured Q_compute) vs Grid Size",
        out_path,
        dpi,
    )


def plot_overcompute_percentage(runs: dict[str, pd.DataFrame], column: str, out_path: Path, dpi: int) -> None:
    """Plot ideal / measured for each run — values > 1 mean measured is below ideal."""

    all_grids = grid_union(runs)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for run_name, df in runs.items():
        if column not in df.columns:
            continue

        cores = df["cores"].to_numpy(dtype=float)
        values = df[column].to_numpy(dtype=float)

        finite = np.where(np.isfinite(values) & (values != 0))[0]
        if finite.size == 0:
            continue

        i0 = int(finite[0])
        c0, y0 = cores[i0], values[i0]
        if c0 == 0 or not np.isfinite(y0) or y0 == 0:
            continue

        ideal = y0 * (cores / c0)
        ratio = ideal / values

        ax.plot(x_positions(df, all_grids), ratio, marker="o", label=run_name)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="Ideal (ratio = 1)")
    finish_and_save(
        fig, ax, all_grids, "Ideal / Measured", "Overcompute Percentage (Ideal / Measured) vs Grid Size", out_path, dpi
    )


def plot_percentage_change(
    runs: dict[str, pd.DataFrame], column: str, ylabel: str, title: str, out_path: Path, dpi: int
) -> None:
    """Plot % change relative to first (minimum) grid: (value[i] - value[0]) / value[0] * 100."""

    all_grids = grid_union(runs)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for run_name, df in runs.items():
        if column not in df.columns:
            continue

        values = df[column].to_numpy(dtype=float)

        finite = np.where(np.isfinite(values) & (values != 0))[0]
        if finite.size == 0:
            continue

        y0 = values[int(finite[0])]
        pct = (values - y0) / y0 * 100.0

        ax.plot(x_positions(df, all_grids), pct, marker="o", label=run_name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    finish_and_save(fig, ax, all_grids, ylabel, title, out_path, dpi)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare power sweep runs from program_intervals CSV files.")
    parser.add_argument(
        "--out-root", type=Path, default=Path("out"), help="Root directory containing run subdirectories. Default: out/"
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        default=[],
        help="Only include runs whose directory name starts with one of these prefixes.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Where to save comparison PNGs. Default: <out-root>/comparison/"
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--no-overcompute",
        action="store_true",
        help=(
            "Skip the overcompute charts. Their ideal line assumes work grows with the core count, "
            "which only holds for fixed-per-core runs; use this for split-mode runs."
        ),
    )
    args = parser.parse_args()

    out_root = args.out_root.expanduser().resolve()
    output_dir = args.output_dir or out_root / "comparison"

    if not out_root.is_dir():
        print(f"ERROR: {out_root} is not a directory.", file=sys.stderr)
        return 1

    print(f"Scanning: {out_root}")
    if args.filter:
        print(f"Filter:   {args.filter}")

    runs = load_runs(out_root, args.filter)
    if not runs:
        print("No matching runs found.")
        return 1

    print(f"Loaded {len(runs)} run(s): {list(runs.keys())}")
    print(f"Output → {output_dir}\n")

    for metric_name, (column, ylabel, title) in METRICS.items():
        # Check column exists in at least one run
        if not any(column in df.columns for df in runs.values()):
            print(f"  SKIP {metric_name}: column '{column}' not found.")
            continue
        plot_metric(
            runs=runs,
            column=column,
            ylabel=ylabel,
            title=title,
            out_path=output_dir / f"{metric_name}.png",
            dpi=args.dpi,
        )

    for col, out_name, ylabel, title in [
        (
            "charge_avg_mah",
            "charge_total_avg_percentage",
            "% change vs min grid",
            "Total Charge Avg — % change vs minimum grid",
        ),
        (
            "charge_peak_mah",
            "charge_total_peak_percentage",
            "% change vs min grid",
            "Total Charge Peak — % change vs minimum grid",
        ),
    ]:
        plot_percentage_change(
            runs=runs,
            column=col,
            ylabel=ylabel,
            title=title,
            out_path=output_dir / f"{out_name}.png",
            dpi=args.dpi,
        )

    if not args.no_overcompute:
        plot_overcompute(
            runs=runs,
            column="dynamic_charge_peak_mah",
            out_path=output_dir / "overcompute.png",
            dpi=args.dpi,
        )
        plot_overcompute_percentage(
            runs=runs,
            column="dynamic_charge_peak_mah",
            out_path=output_dir / "overcompute_percentage.png",
            dpi=args.dpi,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
