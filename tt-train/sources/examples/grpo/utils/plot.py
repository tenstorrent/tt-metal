#!/usr/bin/env python3
"""Generic CSV metric plotter.

Usage:
    plot.py <csv_file> <metric> [<metric2> ...] [--x <x_col>] [--window <n>] [--out <output_png>]

Examples:
    plot.py metrics.csv reward_mean
    plot.py metrics.csv reward_mean reward_std
    plot.py metrics.csv reward_mean --x batch --window 5 --out my_plot.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import csv


def load_csv(path: str) -> dict[str, list[float]]:
    """Load a CSV file and return a dict of column name -> list of values."""
    columns: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                columns.setdefault(key, [])
                try:
                    columns[key].append(float(val))
                except (ValueError, TypeError):
                    columns[key].append(val)
    return columns


def rolling_avg(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (smoothed_values, smoothed_indices) using a simple moving average."""
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    indices = np.arange(window, len(values) + 1)
    return smoothed, indices


def plot_metrics(csv_path: str, metrics: list[str], x_col: str, window: int, out_path: str) -> None:
    data = load_csv(csv_path)

    missing = [m for m in metrics + [x_col] if m not in data]
    if missing:
        available = ", ".join(data.keys())
        print(f"Error: column(s) not found in CSV: {', '.join(missing)}")
        print(f"Available columns: {available}")
        sys.exit(1)

    x = np.array(data[x_col])

    n = len(metrics)
    _, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True, squeeze=False)

    title = os.path.basename(csv_path)
    axes[0][0].set_title(title)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, metric in enumerate(metrics):
        ax = axes[i][0]
        color = colors[i % len(colors)]
        y = np.array(data[metric])

        ax.scatter(x, y, alpha=0.2, s=10, color=color)

        w = min(window, max(1, len(y) // 4))
        if w > 1:
            y_smooth, x_smooth_idx = rolling_avg(y, w)
            x_smooth = x[x_smooth_idx - 1]
            ax.plot(x_smooth, y_smooth, color=color, linewidth=2, label=f"rolling avg (w={w})")
            ax.legend()

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel(x_col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one or more metrics from a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("metrics", nargs="+", help="Column name(s) to plot")
    parser.add_argument("--x", default="step", metavar="COL", help="Column to use as the x-axis (default: step)")
    parser.add_argument("--window", type=int, default=20, help="Rolling-average window size (default: 20)")
    parser.add_argument(
        "--out", default=None, metavar="FILE", help="Output PNG path (default: <csv_stem>_<metrics>.png)"
    )

    args = parser.parse_args()

    if args.out is None:
        stem = os.path.splitext(args.csv_file)[0]
        suffix = "_".join(args.metrics)
        args.out = f"{stem}_{suffix}.png"

    plot_metrics(args.csv_file, args.metrics, args.x, args.window, args.out)


if __name__ == "__main__":
    main()
