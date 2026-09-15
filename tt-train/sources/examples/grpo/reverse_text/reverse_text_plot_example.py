#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Plot the grpo_metrics.csv written by the reverse-text GRPO run.

With no arguments, plots the newest run under
``${TT_METAL_RUNTIME_ROOT}/generated/tt-train/grpo_reverse_text_run/`` and writes
the PNG next to its CSV.

Usage:
    reverse_text_plot_example.py [<csv_file>] [--out <output_png>] [--window <n>]
                                 [--metrics <col> ...]

Examples:
    reverse_text_plot_example.py
    reverse_text_plot_example.py generated/tt-train/grpo_reverse_text_run/*/grpo_metrics.csv
    reverse_text_plot_example.py --metrics reward eval_similarity --window 5
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS_SUBDIR = "generated/tt-train/grpo_reverse_text_run"

# Every column GRPOMonitor writes, in reading order: the learning signal first,
# then what the policy is producing, then the wall-clock cost.
METRICS = [
    ("reward", "reward (similarity ratio)"),
    ("eval_similarity", "eval similarity"),
    ("eval_chars", "eval matched chars"),
    ("eval_format", "eval format rate"),
    ("avg_length", "mean completion (tokens)"),
    ("step_time_s", "step time (s)"),
    ("generation_time_s", "generation time (s)"),
]


def repo_root() -> Path:
    """tt-metal root: the env var when set, else derived from this file's path."""
    env_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if env_root:
        return Path(env_root)
    # .../tt-train/sources/examples/grpo/reverse_text/<this file>
    return Path(__file__).resolve().parents[5]


def latest_metrics_csv() -> str:
    runs_dir = repo_root() / RUNS_SUBDIR
    candidates = list(runs_dir.glob("*/grpo_metrics.csv"))
    if not candidates:
        sys.exit(f"No grpo_metrics.csv found under {runs_dir}; pass a CSV path explicitly.")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def load_csv(path: str) -> dict[str, list]:
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
    """Return (smoothed_values, 1-based end indices) using a simple moving average."""
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    indices = np.arange(window, len(values) + 1)
    return smoothed, indices


def plot_metrics(csv_path: str, metrics: list[tuple[str, str]], window: int, out_path: str) -> None:
    data = load_csv(csv_path)
    if "step" not in data:
        sys.exit(f"Error: no 'step' column in {csv_path}. Columns: {', '.join(data)}")

    missing = [column for column, _ in metrics if column not in data]
    if missing:
        sys.exit(f"Error: column(s) not found in CSV: {', '.join(missing)}. Columns: {', '.join(data)}")

    x = np.array(data["step"])
    n_cols = 2 if len(metrics) > 3 else 1
    n_rows = -(-len(metrics) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3 * n_rows), sharex=True, squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    run_name = Path(csv_path).parent.name
    fig.suptitle(f"reverse-text GRPO — {run_name}")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (column, label) in enumerate(metrics):
        ax = flat_axes[i]
        color = colors[i % len(colors)]
        y = np.array(data[column], dtype=float)

        ax.plot(x, y, marker="o", markersize=3, linewidth=1, alpha=0.4, color=color)

        w = min(window, max(1, len(y) // 4))
        if w > 1:
            y_smooth, end_indices = rolling_avg(y, w)
            ax.plot(x[end_indices - 1], y_smooth, color=color, linewidth=2, label=f"rolling avg (w={w})")
            ax.legend(fontsize="small")

        ax.set_ylabel(label)
        # Anchor at 0 so panel heights read as absolute magnitudes, not as the
        # zoomed-in wiggle autoscale would show.
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # Hide the trailing cell(s) when the metric count does not fill the grid.
    for ax in flat_axes[len(metrics) :]:
        ax.set_visible(False)

    # sharex hides tick labels on every row but the last, so label the lowest
    # visible axis of each column -- which is not the last row when the grid has
    # an empty trailing cell.
    for i, ax in enumerate(flat_axes[: len(metrics)]):
        if i + n_cols >= len(metrics):
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the reverse-text GRPO metrics CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=None,
        help="Path to grpo_metrics.csv (default: the newest reverse-text run under generated/)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="COL",
        help="Columns to plot (default: every column GRPOMonitor writes)",
    )
    parser.add_argument("--window", type=int, default=3, help="Rolling-average window size (default: 3)")
    parser.add_argument("--out", default=None, metavar="FILE", help="Output PNG path (default: next to the CSV)")

    args = parser.parse_args()

    csv_file = args.csv_file or latest_metrics_csv()
    metrics = [(column, column) for column in args.metrics] if args.metrics else METRICS
    out_path = args.out or str(Path(csv_file).with_suffix(".png"))

    plot_metrics(csv_file, metrics, args.window, out_path)


if __name__ == "__main__":
    main()
