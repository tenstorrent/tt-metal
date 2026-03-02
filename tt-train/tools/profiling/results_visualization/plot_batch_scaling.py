#!/usr/bin/env python3
"""Extract batch-size-scaling subset from the full results CSV and produce 5 plots.

The script filters rows where tp==1, dp==1, n_blocks==4,
runner_type==memory_efficient, profiler==naive, and batch_size in {1,2,4,8,16},
and creates:
  - batch_scaling.csv          - the filtered subset
  - ts_vs_batch.png            - tokens/s vs batch size
  - fwd_ms_vs_batch.png        - forward time vs batch size
  - bwd_ms_vs_batch.png        - backward time vs batch size
  - opt_ms_vs_batch.png        - optimizer time vs batch size
  - step_time_ms_vs_batch.png  - total step time vs batch size

Each plot includes a dashed "ideal" line (linear growth from the batch=1
baseline for fwd/bwd/step; constant for optimizer) and a secondary right
y-axis showing MFU %.

Usage:
    python plot_batch_scaling.py <input_csv> <output_dir>

Example:
    python plot_batch_scaling.py experiments/all_results_3_roofline_bw.csv experiments/batch_scaling
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BATCH_VALUES = [1, 2, 4, 8, 16]
N_BLOCKS = 4

PLOT_SPECS = [
    ("t/s", "Tokens / s", "ts_vs_batch.png", "constant", "total_mfu_perc"),
    ("fwd_ms", "Forward time (ms)", "fwd_ms_vs_batch.png", "multiply", "fwd_mfu_perc"),
    ("bwd_ms", "Backward time (ms)", "bwd_ms_vs_batch.png", "multiply", "bwd_mfu_perc"),
    (
        "opt_ms",
        "Optimizer time (ms)",
        "opt_ms_vs_batch.png",
        "constant",
        "opt_mfu_perc",
    ),
    (
        "step_time_ms",
        "Step time (ms)",
        "step_time_ms_vs_batch.png",
        "multiply",
        "total_mfu_perc",
    ),
]

COLOR = "#1f77b4"
MFU_COLOR = "#9467bd"


def filter_batch_scaling(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["tp"] == 1)
        & (df["dp"] == 1)
        & (df["n_blocks"] == N_BLOCKS)
        & df["batch_size"].isin(BATCH_VALUES)
        & (df["runner_type"] == "memory_efficient")
        & (df["profiler"] == "naive")
    )
    subset = df.loc[mask].copy()
    subset.sort_values("batch_size", inplace=True)
    return subset


def _ideal_y(baseline: float, x_arr: np.ndarray, scaling: str) -> np.ndarray:
    if scaling == "multiply":
        return baseline * x_arr
    return np.full_like(x_arr, baseline, dtype=float)


def make_plot(
    subset: pd.DataFrame,
    col: str,
    ylabel: str,
    out_path: Path | None,
    scaling: str,
    mfu_col: str,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grp = subset.sort_values("batch_size")
    has_mfu = mfu_col in grp.columns and grp[mfu_col].notna().any()

    ax.plot(
        grp["batch_size"],
        grp[col],
        marker="o",
        color=COLOR,
        linewidth=2,
        markersize=7,
        label=f"{N_BLOCKS} blocks",
    )

    baseline_row = grp[grp["batch_size"] == 1]
    if not baseline_row.empty:
        base_val = baseline_row[col].iloc[0]
        x_dense = np.linspace(BATCH_VALUES[0], BATCH_VALUES[-1], 200)
        ax.plot(
            x_dense,
            _ideal_y(base_val, x_dense, scaling),
            color="red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            label="ideal",
        )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  vs  Batch size")
    ax.set_xticks(BATCH_VALUES)
    ax.grid(True, alpha=0.3)

    if has_mfu:
        ax2 = ax.twinx()
        valid = grp.dropna(subset=[mfu_col])
        if not valid.empty:
            ax2.plot(
                valid["batch_size"],
                valid[mfu_col],
                marker="x",
                color=MFU_COLOR,
                linewidth=1.2,
                markersize=8,
                alpha=0.7,
                label="MFU %",
            )
        ax2.set_ylabel("MFU %", color=MFU_COLOR)
        ax2.tick_params(axis="y", labelcolor=MFU_COLOR)
        ax2.set_ylim(0, 100)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax.legend()

    fig.tight_layout()
    if show:
        plt.show()
    elif out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}")


def plot_all(
    df: pd.DataFrame, output_dir: Path | None = None, show: bool = False
) -> pd.DataFrame:
    """Filter and plot all batch scaling charts. Returns the filtered subset."""
    subset = filter_batch_scaling(df)
    if subset.empty:
        print("No batch-scaling rows found.")
        return subset
    for col, ylabel, fname, scaling, mfu_col in PLOT_SPECS:
        out = output_dir / fname if output_dir else None
        make_plot(subset, col, ylabel, out, scaling, mfu_col, show=show)
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot batch-size-scaling results.")
    parser.add_argument("input_csv", type=Path, help="Full results CSV")
    parser.add_argument("output_dir", type=Path, help="Directory for outputs")
    args = parser.parse_args()

    if not args.input_csv.exists():
        sys.exit(f"Input file not found: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    subset = plot_all(df, output_dir=args.output_dir)

    if not subset.empty:
        csv_out = args.output_dir / "batch_scaling.csv"
        subset.to_csv(csv_out, index=False)
        print(f"Wrote {len(subset)} rows to {csv_out}")

    print("Done.")


if __name__ == "__main__":
    main()
