#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Plot LB 8x1 (no contention) vs Galaxy 8x4 (with contention) per-col, per-op times
across all MoE layers. Produces 8 PNG bar plots:
  - 2 ops (Dispatch, Combine) x 4 dispatch groups (col0..3) = 8 plots
  - Per layer: side-by-side blue (LB) and orange (GLX) bars
  - Spacing between layers so labels are readable

Usage:
    python plot_lb_vs_glx.py \\
        --lb-summary  /data/nmilicevic/tt-metal/lb_summary_linear-8-2link.csv \\
        --glx-summary /data/nmilicevic/glx_8x4_dispatch_combine_results_full/summary_mesh-8x4-2link.csv \\
        --out-dir     /data/nmilicevic/tt-metal/lb_vs_glx_plots
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless backend, just write PNGs
import matplotlib.pyplot as plt

# Map LB summary column names → normalized op_col format
_LB_RENAME = {
    "dispatch_col0_max_chip_med_ns": "dispatch_col0",
    "dispatch_col1_max_chip_med_ns": "dispatch_col1",
    "dispatch_col2_max_chip_med_ns": "dispatch_col2",
    "dispatch_col3_max_chip_med_ns": "dispatch_col3",
    "dispatch_layer_max_ns": "dispatch_max",
    "combine_col0_max_chip_med_ns": "combine_col0",
    "combine_col1_max_chip_med_ns": "combine_col1",
    "combine_col2_max_chip_med_ns": "combine_col2",
    "combine_col3_max_chip_med_ns": "combine_col3",
    "combine_layer_max_ns": "combine_max",
}
# Map GLX summary column names → normalized
_GLX_RENAME = {
    "dispatch_col0_ns": "dispatch_col0",
    "dispatch_col1_ns": "dispatch_col1",
    "dispatch_col2_ns": "dispatch_col2",
    "dispatch_col3_ns": "dispatch_col3",
    "dispatch_max_ns": "dispatch_max",
    "combine_col0_ns": "combine_col0",
    "combine_col1_ns": "combine_col1",
    "combine_col2_ns": "combine_col2",
    "combine_col3_ns": "combine_col3",
    "combine_max_ns": "combine_max",
}


def _load(path: Path, rename: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=rename)
    return df.set_index("layer_idx").sort_index()


def _draw_bars(ax, layers, lb_vals_ms, glx_vals_ms, title, xlabel=True):
    n = len(layers)
    bar_w = 0.4
    x = np.arange(n)
    x_lb = x - bar_w / 2 - 0.02
    x_glx = x + bar_w / 2 + 0.02

    ax.bar(x_lb, lb_vals_ms, width=bar_w, color="tab:blue", label="LB 8x1 (200G)")
    ax.bar(x_glx, glx_vals_ms, width=bar_w, color="tab:orange", label="Galaxy 8x4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L:02d}" for L in layers], rotation=70, fontsize=7)
    if xlabel:
        ax.set_xlabel("MoE layer")
    ax.set_ylabel("kernel dur (ms)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.7, n - 0.3)


def plot_one(lb: pd.DataFrame, glx: pd.DataFrame, op: str, col_or_max: int | str, out_path: Path) -> None:
    if col_or_max == "max":
        key = f"{op}_max"
        title_suffix = "max across dispatch groups"
    else:
        key = f"{op}_col{col_or_max}"
        title_suffix = f"group {col_or_max}"
    layers = sorted(set(lb.index) & set(glx.index))
    lb_vals_ms = [lb.loc[L, key] / 1e6 for L in layers]
    glx_vals_ms = [glx.loc[L, key] / 1e6 for L in layers]

    fig, ax = plt.subplots(figsize=(max(14, len(layers) * 0.28), 5))
    _draw_bars(ax, layers, lb_vals_ms, glx_vals_ms, f"{op.capitalize()} — {title_suffix}")
    if col_or_max == "max":
        avg_lb = sum(lb_vals_ms) / len(lb_vals_ms)
        avg_glx = sum(glx_vals_ms) / len(glx_vals_ms)
        ax.axhline(avg_lb, color="tab:blue", linestyle="--", linewidth=1.4, label=f"LB avg = {avg_lb:.2f} ms", zorder=5)
        ax.axhline(
            avg_glx, color="tab:orange", linestyle="--", linewidth=1.4, label=f"Galaxy avg = {avg_glx:.2f} ms", zorder=5
        )
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_grid(lb: pd.DataFrame, glx: pd.DataFrame, out_path: Path) -> None:
    """Combined 3-row grid:
      row 0: Dispatch group 0..3
      row 1: Combine  group 0..3
      row 2: Dispatch max, Combine max (each spans 2 cells for balance)
    A thin rectangle is drawn around each column (group), enclosing the dispatch+combine
    pair so the 4 dispatch groups are visually separated.
    """
    from matplotlib.patches import Rectangle

    layers = sorted(set(lb.index) & set(glx.index))
    ops = ["dispatch", "combine"]
    fig_w = max(14, len(layers) * 0.32) * 2
    fig = plt.figure(figsize=(fig_w, 13.5))
    # Tighter horizontal spacing → subplots take more width. Slightly more vertical hspace so
    # the row-0 x-axis labels still have breathing room below.
    gs = fig.add_gridspec(3, 4, hspace=0.65, wspace=0.10)

    # Track the per-col subplot pairs so we can outline each group after layout.
    group_axes: dict[int, list] = {ci: [] for ci in range(4)}

    # Rows 0-1: per-col
    for ri, op in enumerate(ops):
        row_axes = []
        for ci in range(4):
            ax = fig.add_subplot(gs[ri, ci], sharey=row_axes[0] if row_axes else None)
            row_axes.append(ax)
            group_axes[ci].append(ax)
            key = f"{op}_col{ci}"
            lb_vals_ms = [lb.loc[L, key] / 1e6 for L in layers]
            glx_vals_ms = [glx.loc[L, key] / 1e6 for L in layers]
            _draw_bars(ax, layers, lb_vals_ms, glx_vals_ms, f"{op.capitalize()} — group {ci}", xlabel=False)
            if ci > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    # Row 2: max plots, each spanning 2 cells. Add a horizontal avg line for each machine.
    ax_disp_max = fig.add_subplot(gs[2, 0:2])
    ax_comb_max = fig.add_subplot(gs[2, 2:4])
    for ax, op in [(ax_disp_max, "dispatch"), (ax_comb_max, "combine")]:
        key = f"{op}_max"
        lb_vals_ms = [lb.loc[L, key] / 1e6 for L in layers]
        glx_vals_ms = [glx.loc[L, key] / 1e6 for L in layers]
        _draw_bars(ax, layers, lb_vals_ms, glx_vals_ms, f"{op.capitalize()} — max", xlabel=True)
        avg_lb = sum(lb_vals_ms) / len(lb_vals_ms)
        avg_glx = sum(glx_vals_ms) / len(glx_vals_ms)
        ax.axhline(avg_lb, color="tab:blue", linestyle="--", linewidth=1.4, label=f"LB avg = {avg_lb:.2f} ms", zorder=5)
        ax.axhline(
            avg_glx, color="tab:orange", linestyle="--", linewidth=1.4, label=f"Galaxy avg = {avg_glx:.2f} ms", zorder=5
        )
        # Refresh legend now that lines were added (the bars were already in legend).
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("LB 8x1 vs Galaxy 8x4 — Dispatch & Combine per dispatch group, per MoE layer", fontsize=14, y=0.995)
    # Finalize layout before reading axes positions
    fig.canvas.draw()

    # Draw a thin rectangle around each dispatch group's (dispatch, combine) subplot pair.
    # The bottom extends well below the bottom subplot to clear the rotated x-axis tick
    # labels (L03..L60) and the "MoE layer" xlabel. Side padding is tight so groups sit
    # close together visually.
    pad_x = 0.003
    pad_y_top = 0.020  # extra above row-0 title
    pad_y_bottom = 0.060  # well below row-1 x-axis labels + xlabel
    for ci, axes_pair in group_axes.items():
        ax_top, ax_bot = axes_pair  # row 0 (dispatch), row 1 (combine)
        b_top = ax_top.get_position()
        b_bot = ax_bot.get_position()
        x0 = min(b_top.x0, b_bot.x0) - pad_x
        y0 = b_bot.y0 - pad_y_bottom
        x1 = max(b_top.x1, b_bot.x1) + pad_x
        y1 = b_top.y1 + pad_y_top
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.2,
            edgecolor="black",
            facecolor="none",
            transform=fig.transFigure,
            zorder=10,
            clip_on=False,
        )
        fig.add_artist(rect)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lb-summary", required=True, help="LB summary CSV (summary_linear-8-2link.csv)")
    ap.add_argument("--glx-summary", required=True, help="Galaxy 8x4 summary CSV (summary_mesh-8x4-2link.csv)")
    ap.add_argument("--out-dir", required=True, help="Where to write the 8 PNGs")
    args = ap.parse_args()

    lb = _load(Path(args.lb_summary), _LB_RENAME)
    glx = _load(Path(args.glx_summary), _GLX_RENAME)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for op in ["dispatch", "combine"]:
        for col in range(4):
            plot_one(lb, glx, op, col, out_dir / f"{op}_col{col}.png")
        plot_one(lb, glx, op, "max", out_dir / f"{op}_max.png")
    plot_grid(lb, glx, out_dir / "all_lb_vs_glx.png")

    print(f"\nDone. 10 standalone + 1 combined PNG in {out_dir}")


if __name__ == "__main__":
    main()
