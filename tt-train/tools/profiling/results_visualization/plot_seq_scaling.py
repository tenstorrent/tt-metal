# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Extract sequence-length-scaling subset from the full results CSV and produce 6 plots.

The script filters rows where tp==1, dp==1, n_blocks==22,
runner_type==memory_efficient, profiler==naive, and seq_len in
{512, 1024, 2048, 4096, 8192, 16384}, and creates:
  - seq_scaling.csv            - the filtered subset
  - ts_vs_seq.png              - tokens/s vs sequence length
  - fwd_ms_vs_seq.png          - forward time vs sequence length
  - bwd_ms_vs_seq.png          - backward time vs sequence length
  - opt_ms_vs_seq.png          - optimizer time vs sequence length
  - step_time_ms_vs_seq.png    - total step time vs sequence length
  - dram_peak_vs_seq.png       - DRAM peak usage vs sequence length

Each plot includes a dashed "ideal" line (linear scaling from the smallest
seq_len baseline for fwd/bwd/step/memory; constant for optimizer and tokens/s).
The x-axis uses log2 scale since sequence lengths are powers of 2.

Usage:
    python plot_seq_scaling.py <input_csv> <output_dir>

Example:
    python plot_seq_scaling.py experiments/all_results.csv experiments/seq_scaling
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SEQ_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_N_BLOCKS = 22

PLOT_SPECS = [
    ("t/s", "Tokens / s", "ts_vs_seq.png", "constant", "total_mfu_perc"),
    ("fwd_ms", "Forward time (ms)", "fwd_ms_vs_seq.png", "multiply", "fwd_mfu_perc"),
    ("bwd_ms", "Backward time (ms)", "bwd_ms_vs_seq.png", "multiply", "bwd_mfu_perc"),
    ("opt_ms", "Optimizer time (ms)", "opt_ms_vs_seq.png", "constant", "opt_mfu_perc"),
    (
        "step_time_ms",
        "Step time (ms)",
        "step_time_ms_vs_seq.png",
        "multiply",
        "total_mfu_perc",
    ),
    (
        "dram_peak_mb",
        "DRAM Peak (MB)",
        "dram_peak_vs_seq.png",
        "multiply",
        None,
    ),
]

COLOR = "#1f77b4"
MFU_COLOR = "#9467bd"


def filter_seq_scaling(df: pd.DataFrame, n_blocks=None) -> pd.DataFrame:
    n_blocks = n_blocks or DEFAULT_N_BLOCKS
    has_seq = "seq_len" in df.columns and df["seq_len"].notna().any()
    if not has_seq:
        return pd.DataFrame()
    mask = (
        (df["tp"] == 1)
        & (df["dp"] == 1)
        & (df["n_blocks"] == n_blocks)
        & df["seq_len"].isin(SEQ_VALUES)
        & (df["runner_type"] == "memory_efficient")
        & (df["profiler"] == "naive")
    )
    subset = df.loc[mask].copy()
    subset.sort_values("seq_len", inplace=True)
    return subset


def _ideal_y(baseline: float, x_arr: np.ndarray, x_base: float, scaling: str) -> np.ndarray:
    """Compute ideal-scaling y values relative to the baseline at x_base."""
    if scaling == "multiply":
        return baseline * (x_arr / x_base)
    return np.full_like(x_arr, baseline, dtype=float)


def make_plot(
    subset: pd.DataFrame,
    col: str,
    ylabel: str,
    out_path: Path | None,
    scaling: str,
    mfu_col: str | None,
    show: bool = False,
) -> None:
    valid_data = subset.dropna(subset=[col])
    if valid_data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    grp = valid_data.sort_values("seq_len")
    has_mfu = mfu_col is not None and mfu_col in grp.columns and grp[mfu_col].notna().any()

    ax.plot(
        grp["seq_len"],
        grp[col],
        marker="o",
        color=COLOR,
        linewidth=2,
        markersize=7,
        label=f"{valid_data['n_blocks'].iloc[0]} blocks",
    )

    seq_min = grp["seq_len"].min()
    baseline_row = grp[grp["seq_len"] == seq_min]
    if not baseline_row.empty:
        base_val = baseline_row[col].iloc[0]
        x_dense = np.geomspace(SEQ_VALUES[0], SEQ_VALUES[-1], 200)
        ax.plot(
            x_dense,
            _ideal_y(base_val, x_dense, seq_min, scaling),
            color="red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            label="ideal (linear)",
        )

    ax.set_xscale("log", base=2)
    present = sorted(grp["seq_len"].unique())
    ax.set_xticks(present)
    ax.set_xticklabels([str(int(v)) for v in present])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  vs  Sequence length")
    ax.grid(True, alpha=0.3)

    if has_mfu:
        ax2 = ax.twinx()
        mfu_valid = grp.dropna(subset=[mfu_col])
        if not mfu_valid.empty:
            ax2.plot(
                mfu_valid["seq_len"],
                mfu_valid[mfu_col],
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
    df: pd.DataFrame,
    output_dir: Path | None = None,
    show: bool = False,
    n_blocks=None,
) -> pd.DataFrame:
    """Filter and plot all sequence length scaling charts. Returns the filtered subset."""
    subset = filter_seq_scaling(df, n_blocks)
    if subset.empty:
        print("No seq-len-scaling rows found.")
        return subset
    for col, ylabel, fname, scaling, mfu_col in PLOT_SPECS:
        out = output_dir / fname if output_dir else None
        make_plot(subset, col, ylabel, out, scaling, mfu_col, show=show)
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sequence-length-scaling results.")
    parser.add_argument("input_csv", type=Path, help="Full results CSV")
    parser.add_argument("output_dir", type=Path, help="Directory for outputs")
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=DEFAULT_N_BLOCKS,
        help=f"Block count to filter on (default: {DEFAULT_N_BLOCKS})",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        sys.exit(f"Input file not found: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    subset = plot_all(df, output_dir=args.output_dir, n_blocks=args.n_blocks)

    if not subset.empty:
        csv_out = args.output_dir / "seq_scaling.csv"
        subset.to_csv(csv_out, index=False)
        print(f"Wrote {len(subset)} rows to {csv_out}")

    print("Done.")


if __name__ == "__main__":
    main()
