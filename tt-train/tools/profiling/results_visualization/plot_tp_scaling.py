#!/usr/bin/env python3
"""Extract TP-scaling subset from the full results CSV and produce 6 plots.

The script filters rows where dp==1 and tp in {1, 2, 4, 8} (pure TP scaling),
groups them by n_blocks, and creates:
  - tp_scaling.csv          – the filtered subset
  - ts_vs_tp.png            – tokens/s vs TP
  - tsd_vs_tp.png           – tokens/s/device vs TP
  - fwd_ms_vs_tp.png        – forward time vs TP
  - bwd_ms_vs_tp.png        – backward time vs TP
  - opt_ms_vs_tp.png        – optimizer time vs TP
  - step_time_ms_vs_tp.png  – total step time vs TP

Each plot includes a red dashed "ideal scaling" line derived from the tp=1
baseline for each block count.

Usage:
    python plot_tp_scaling.py <input_csv> <output_dir>

Example:
    python plot_tp_scaling.py experiments/all_results_3_roofline_bw.csv experiments/tp_scaling
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TP_VALUES = [1, 2, 4, 8]
BLOCK_VALUES = [2, 4, 8]

PLOT_SPECS = [
    ("t/s", "Tokens / s", "ts_vs_tp.png", "multiply", None),
    ("t/s/d", "Tokens / s / device", "tsd_vs_tp.png", "constant", None),
    ("fwd_ms", "Forward time (ms)", "fwd_ms_vs_tp.png", "divide", "fwd_ccl_ms"),
    ("bwd_ms", "Backward time (ms)", "bwd_ms_vs_tp.png", "divide", "bwd_ccl_ms"),
    ("opt_ms", "Optimizer time (ms)", "opt_ms_vs_tp.png", "divide", "opt_ccl_ms"),
    (
        "step_time_ms",
        "Step time (ms)",
        "step_time_ms_vs_tp.png",
        "divide",
        "total_ccl_ms",
    ),
]

COLORS = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}
MARKERS = {2: "o", 4: "s", 8: "^"}
CCL_COLOR = "#d62728"


def filter_tp_scaling(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["dp"] == 1)
        & df["tp"].isin(TP_VALUES)
        & df["n_blocks"].isin(BLOCK_VALUES)
        & (df["runner_type"] == "default")
        & (df["profiler"] == "tracy")
    )
    subset = df.loc[mask].copy()
    subset.sort_values(["n_blocks", "tp"], inplace=True)
    return subset


def _ideal_y(baseline: float, tp_arr: np.ndarray, scaling: str) -> np.ndarray:
    """Compute ideal-scaling y values from the tp=1 baseline."""
    if scaling == "multiply":
        return baseline * tp_arr
    if scaling == "constant":
        return np.full_like(tp_arr, baseline, dtype=float)
    return baseline / tp_arr


MFU_COL = "total_mfu_perc"
MFU_COLOR = "#9467bd"


def make_plot(
    subset: pd.DataFrame,
    col: str,
    ylabel: str,
    out_path: Path,
    scaling: str,
    ccl_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tp_dense = np.linspace(TP_VALUES[0], TP_VALUES[-1], 200)
    ideal_drawn = False
    has_mfu = MFU_COL in subset.columns and subset[MFU_COL].notna().any()
    has_ccl = (
        ccl_col is not None
        and ccl_col in subset.columns
        and subset[ccl_col].notna().any()
    )

    for blk in sorted(subset["n_blocks"].unique()):
        grp = subset[subset["n_blocks"] == blk].sort_values("tp")
        ax.plot(
            grp["tp"],
            grp[col],
            marker=MARKERS.get(blk, "o"),
            color=COLORS.get(blk, None),
            linewidth=2,
            markersize=7,
            label=f"{blk} blocks",
        )

        baseline_row = grp[grp["tp"] == 1]
        if not baseline_row.empty:
            base_val = baseline_row[col].iloc[0]
            ideal_label = "ideal" if not ideal_drawn else None
            ax.plot(
                tp_dense,
                _ideal_y(base_val, tp_dense, scaling),
                color=COLORS.get(blk, None),
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
                label=ideal_label,
            )
            ideal_drawn = True

    if has_ccl:
        ccl_drawn = False
        for blk in sorted(subset["n_blocks"].unique()):
            grp = subset[subset["n_blocks"] == blk].sort_values("tp")
            valid = grp.dropna(subset=[ccl_col])
            if not valid.empty:
                lbl = "CCL time" if not ccl_drawn else None
                ax.plot(
                    valid["tp"],
                    valid[ccl_col],
                    marker="d",
                    color=COLORS.get(blk, None),
                    linewidth=1.2,
                    markersize=6,
                    alpha=0.7,
                    linestyle=":",
                    label=lbl,
                )
                ccl_drawn = True

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Tensor Parallelism (TP)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  vs  TP")
    ax.set_xticks(TP_VALUES)
    ax.grid(True, alpha=0.3)

    if has_mfu:
        ax2 = ax.twinx()
        mfu_drawn = False
        for blk in sorted(subset["n_blocks"].unique()):
            grp = subset[subset["n_blocks"] == blk].sort_values("tp")
            valid = grp.dropna(subset=[MFU_COL])
            if not valid.empty:
                lbl = "MFU %" if not mfu_drawn else None
                ax2.plot(
                    valid["tp"],
                    valid[MFU_COL],
                    marker="x",
                    color=MFU_COLOR,
                    linewidth=1.2,
                    markersize=8,
                    alpha=0.7,
                    label=lbl,
                )
                mfu_drawn = True
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
    parser = argparse.ArgumentParser(description="Plot TP-scaling results.")
    parser.add_argument("input_csv", type=Path, help="Full results CSV")
    parser.add_argument("output_dir", type=Path, help="Directory for outputs")
    args = parser.parse_args()

    if not args.input_csv.exists():
        sys.exit(f"Input file not found: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    subset = filter_tp_scaling(df)

    if subset.empty:
        sys.exit(
            "No TP-scaling rows found (dp==1, tp in {1,2,4,8}, n_blocks in {2,4,8})."
        )

    csv_out = args.output_dir / "tp_scaling.csv"
    subset.to_csv(csv_out, index=False)
    print(f"Wrote {len(subset)} rows to {csv_out}")

    for col, ylabel, fname, scaling, ccl_col in PLOT_SPECS:
        make_plot(subset, col, ylabel, args.output_dir / fname, scaling, ccl_col)

    print("Done.")


if __name__ == "__main__":
    main()
