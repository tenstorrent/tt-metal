#!/usr/bin/env python3
"""Extract DDP-scaling subset from the full results CSV and produce 4 plots.

The script filters rows where tp==1, n_blocks==8, and dp in {1, 2, 4, 8, 32}
(pure DDP scaling with naive profiler), and creates:
  - ddp_scaling.csv           – the filtered subset
  - ts_vs_dp.png              – tokens/s vs DDP
  - tsd_vs_dp.png             – tokens/s/device vs DDP
  - step_time_ms_vs_dp.png    – total step time vs DDP
  - grad_sync_ms_vs_dp.png    – gradient sync time vs DDP

Each plot (except grad_sync) includes a dashed "ideal scaling" line derived
from the dp=1 baseline.

Usage:
    python plot_ddp_scaling.py <input_csv> <output_dir>

Example:
    python plot_ddp_scaling.py experiments/all_results_3_roofline_bw.csv experiments/ddp_scaling
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DP_VALUES = [1, 2, 4, 8, 32]
N_BLOCKS = 8

PLOT_SPECS = [
    ("t/s", "Tokens / s", "ts_vs_dp.png", "multiply"),
    ("t/s/d", "Tokens / s / device", "tsd_vs_dp.png", "constant"),
    ("step_time_ms", "Step time (ms)", "step_time_ms_vs_dp.png", "constant"),
    ("grad_sync_ms", "Gradient sync time (ms)", "grad_sync_ms_vs_dp.png", None),
]

COLOR = "#1f77b4"
MFU_COL = "total_mfu_perc"
MFU_COLOR = "#9467bd"


def filter_ddp_scaling(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["tp"] == 1)
        & (df["n_blocks"] == N_BLOCKS)
        & df["dp"].isin(DP_VALUES)
        & (df["runner_type"] == "default")
        & (df["profiler"] == "naive")
    )
    subset = df.loc[mask].copy()
    subset.sort_values("dp", inplace=True)
    return subset


def _ideal_y(baseline: float, dp_arr: np.ndarray, scaling: str) -> np.ndarray:
    if scaling == "multiply":
        return baseline * dp_arr
    return np.full_like(dp_arr, baseline, dtype=float)


def make_plot(
    subset: pd.DataFrame, col: str, ylabel: str, out_path: Path, scaling: str | None
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grp = subset.sort_values("dp")
    has_mfu = MFU_COL in grp.columns and grp[MFU_COL].notna().any()

    ax.plot(
        grp["dp"],
        grp[col],
        marker="o",
        color=COLOR,
        linewidth=2,
        markersize=7,
        label=f"{N_BLOCKS} blocks",
    )

    if scaling is not None:
        baseline_row = grp[grp["dp"] == 1]
        if not baseline_row.empty:
            base_val = baseline_row[col].iloc[0]
            dp_dense = np.linspace(DP_VALUES[0], DP_VALUES[-1], 200)
            ax.plot(
                dp_dense,
                _ideal_y(base_val, dp_dense, scaling),
                color="red",
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label="ideal",
            )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Data Parallelism (DDP)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  vs  DDP")
    ax.set_xticks(DP_VALUES)
    ax.grid(True, alpha=0.3)

    if has_mfu:
        ax2 = ax.twinx()
        valid = grp.dropna(subset=[MFU_COL])
        if not valid.empty:
            ax2.plot(
                valid["dp"],
                valid[MFU_COL],
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DDP-scaling results.")
    parser.add_argument("input_csv", type=Path, help="Full results CSV")
    parser.add_argument("output_dir", type=Path, help="Directory for outputs")
    args = parser.parse_args()

    if not args.input_csv.exists():
        sys.exit(f"Input file not found: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    subset = filter_ddp_scaling(df)

    if subset.empty:
        sys.exit(
            f"No DDP-scaling rows found (tp==1, n_blocks=={N_BLOCKS}, "
            f"dp in {DP_VALUES}, runner=default, profiler=naive)."
        )

    csv_out = args.output_dir / "ddp_scaling.csv"
    subset.to_csv(csv_out, index=False)
    print(f"Wrote {len(subset)} rows to {csv_out}")

    for col, ylabel, fname, scaling in PLOT_SPECS:
        make_plot(subset, col, ylabel, args.output_dir / fname, scaling)

    print("Done.")


if __name__ == "__main__":
    main()
