#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-dimension utilization slice plots — one PNG per (device, dtype, mode, fidelity, sweep).

Each PNG has a 2x2 grid ({host, device} x {no-trace, trace}).  Every fixed-dimension pair
in the sweep is drawn as its own line (color + marker); the legend sits to the right of
all subplots.

Output layout (under images/):
  by-k/{device}-util-{dtype}-{mode}.png  — x = K, one line per (M, N)
  by-m/{device}-util-{dtype}-{mode}.png  — x = M, one line per (K, N)
  by-n/{device}-util-{dtype}-{mode}.png  — x = N, one line per (M, K)

Reads tech_reports/GEMM_FLOPS/data/{bh,wh}.csv
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_GEMM_FLOPS_DIR = Path(__file__).resolve().parent
if str(_GEMM_FLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMM_FLOPS_DIR))
from benchmark_modes import MODE_DISPLAY, add_shape_column, normalize_modes

DATA_DIR = _GEMM_FLOPS_DIR / "data"
IMG_DIR = _GEMM_FLOPS_DIR / "images"

DEVICE_MAP = {
    "bh": "P150 (Blackhole)",
    "wh": "N150 (Wormhole)",
}

DTYPE_MAP = {
    "BFLOAT16": "bf16",
    "BFLOAT8_B": "bf8_b",
    "BFLOAT4_B": "bf4_b",
}

DTYPE_DISPLAY = {
    "bf16": "BFloat16",
    "bf8_b": "BFloat8_B",
    "bf4_b": "BFloat4_B",
}

MARKER_CYCLE = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "X", "P"]

SWEEP_CONFIGS = [
    {
        "subdir": "by-k",
        "x_col": "k",
        "x_label": "K (inner / reduction dim)",
        "fixed_cols": ("m", "n"),
        "legend_fmt": "m={m}, n={n}",
    },
    {
        "subdir": "by-m",
        "x_col": "m",
        "x_label": "M (rows)",
        "fixed_cols": ("k", "n"),
        "legend_fmt": "k={k}, n={n}",
    },
    {
        "subdir": "by-n",
        "x_col": "n",
        "x_label": "N (columns)",
        "fixed_cols": ("m", "k"),
        "legend_fmt": "m={m}, k={k}",
    },
]


def _read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _parse_dtype(raw):
    full = str(raw).split(".")[-1]
    return DTYPE_MAP.get(full, full.lower())


def _parse_fidelity(raw):
    return str(raw).split(".")[-1]


def _find_util_col(df, kind, grid_keyword="user selected grid"):
    prefix = f"{kind} based utilization"
    for col in df.columns:
        if col.startswith(prefix) and grid_keyword in col:
            if df[col].notna().any():
                return col
    return None


def _load_csv(path):
    df = _read_csv(path)
    if df.empty:
        return df
    df["dtype_short"] = df["dtype"].apply(_parse_dtype)
    df["fidelity"] = df["math_fidelity"].apply(_parse_fidelity)
    if df["use_trace"].dtype == object:
        df["use_trace"] = df["use_trace"].astype(str).str.lower() == "true"
    df = add_shape_column(df)
    return normalize_modes(df)


def _group_key_tuple(group_key, n_cols):
    if n_cols == 1:
        return (group_key,)
    return group_key


def _line_points(slice_df, util_col, x_col):
    slice_df = slice_df.dropna(subset=[util_col])
    if slice_df.empty:
        return pd.DataFrame()
    best_rows = []
    for _, group in slice_df.groupby(x_col, sort=True):
        best_rows.append(group.loc[group[util_col].idxmax()])
    return pd.DataFrame(best_rows).sort_values(x_col)


def _eligible_fixed_pairs(df, x_col, fixed_cols):
    """Return fixed-dimension pairs that have at least two distinct x values."""
    pairs = []
    for group_key, slice_df in df.groupby(list(fixed_cols), sort=True):
        if slice_df[x_col].nunique() < 2:
            continue
        key = _group_key_tuple(group_key, len(fixed_cols))
        pairs.append(dict(zip(fixed_cols, key)))
    return pairs


def _style_for_pairs(n_pairs):
    cmap = plt.colormaps["turbo"]
    colors = [cmap(i / max(n_pairs - 1, 1)) for i in range(n_pairs)]
    markers = [MARKER_CYCLE[i % len(MARKER_CYCLE)] for i in range(n_pairs)]
    return colors, markers


def _plot_sweep(
    df,
    dtype_short,
    device_prefix,
    device_label,
    mode,
    fidelity,
    sweep_cfg,
):
    x_col = sweep_cfg["x_col"]
    fixed_cols = sweep_cfg["fixed_cols"]
    fixed_pairs = _eligible_fixed_pairs(df, x_col, fixed_cols)
    if not fixed_pairs:
        return

    host_col = _find_util_col(df, "Host")
    device_col = _find_util_col(df, "Device")
    colors, markers = _style_for_pairs(len(fixed_pairs))

    # Wider figure + right margin for external legend.
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True, sharex=True)
    legend_handles = []

    panel_defs = [
        (0, 0, "Host", False, host_col),
        (0, 1, "Host", True, host_col),
        (1, 0, "Device", False, device_col),
        (1, 1, "Device", True, device_col),
    ]

    for pair_idx, fixed_kwargs in enumerate(fixed_pairs):
        color = colors[pair_idx]
        marker = markers[pair_idx]
        label = sweep_cfg["legend_fmt"].format(**fixed_kwargs)

        mask = np.ones(len(df), dtype=bool)
        for col, val in fixed_kwargs.items():
            mask &= df[col] == val
        pair_df = df[mask]
        if pair_df.empty:
            continue

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.8,
                linestyle="-",
                marker=marker,
                markersize=5,
                markerfacecolor=color,
                markeredgecolor=color,
                label=label,
            )
        )

        for row, col_idx, util_kind, use_trace, util_col_name in panel_defs:
            ax = axes[row][col_idx]
            if util_col_name is None:
                continue

            subset = pair_df[pair_df["use_trace"] == use_trace]
            points = _line_points(subset, util_col_name, x_col)
            if points.empty:
                continue

            ax.plot(
                points[x_col],
                points[util_col_name],
                color=color,
                marker=marker,
                linestyle="-",
                markersize=5,
                markerfacecolor=color,
                markeredgecolor=color,
                linewidth=1.8,
                alpha=0.9,
            )

    for row, col_idx, util_kind, use_trace, util_col_name in panel_defs:
        ax = axes[row][col_idx]
        if util_col_name is None:
            ax.text(
                0.5,
                0.5,
                f"No {util_kind} data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
        trace_label = "With Trace" if use_trace else "Without Trace"
        ax.set_title(f"{util_kind} | {trace_label}", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(0, 110)

    axes[1][0].set_xlabel(sweep_cfg["x_label"], fontsize=11, fontweight="bold")
    axes[1][1].set_xlabel(sweep_cfg["x_label"], fontsize=11, fontweight="bold")
    axes[0][0].set_ylabel("Utilization (%)", fontsize=11, fontweight="bold")
    axes[1][0].set_ylabel("Utilization (%)", fontsize=11, fontweight="bold")

    mode_display = MODE_DISPLAY.get(mode, mode)
    dtype_display = DTYPE_DISPLAY.get(dtype_short, dtype_short)
    fig.suptitle(
        f"Matmul Utilization vs {x_col.upper()} — {device_label} — {dtype_display}\n" f"{mode_display}, {fidelity}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    legend_fontsize = 6 if len(legend_handles) > 40 else 7
    if len(legend_handles) > 120:
        legend_fontsize = 5

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=legend_fontsize,
        framealpha=0.95,
        edgecolor="black",
        handlelength=2.0,
        borderaxespad=0.0,
    )

    out_dir = IMG_DIR / sweep_cfg["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{device_prefix}-util-{dtype_short}-{mode}.png"

    fig.subplots_adjust(top=0.90, right=0.78, wspace=0.22, hspace=0.28)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {out_path} ({len(fixed_pairs)} lines)")
    plt.close()


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    for device_prefix, device_label in DEVICE_MAP.items():
        csv_path = DATA_DIR / f"{device_prefix}.csv"
        df = _load_csv(csv_path)
        if df.empty:
            print(f"No data for {device_label} ({csv_path}) — skipping.")
            continue

        print(f"{device_label}:")
        for dtype_short in sorted(df["dtype_short"].unique()):
            df_dtype = df[df["dtype_short"] == dtype_short].copy()
            print(f"  {dtype_short}:")
            for mode in sorted(df_dtype["mode"].unique()):
                for fidelity in sorted(df_dtype["fidelity"].unique()):
                    subset = df_dtype[(df_dtype["mode"] == mode) & (df_dtype["fidelity"] == fidelity)]
                    if subset.empty:
                        continue
                    for sweep_cfg in SWEEP_CONFIGS:
                        _plot_sweep(
                            subset,
                            dtype_short,
                            device_prefix,
                            device_label,
                            mode,
                            fidelity,
                            sweep_cfg,
                        )


if __name__ == "__main__":
    main()
