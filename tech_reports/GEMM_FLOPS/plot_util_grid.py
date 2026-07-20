#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate per-dtype 2x2 utilization grid plots from a single benchmark CSV.

One PNG per (device, dtype): {device}-util-{dtype}.png
  - x-axis = total matrix elements (M × K × N)
  - line color -> mode, marker -> math fidelity

Reads tech_reports/GEMM_FLOPS/data/{bh,wh}.csv
For per-dimension slice plots (K/M/N sweeps), see plot_util_by_dim.py.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

_GEMM_FLOPS_DIR = Path(__file__).resolve().parent
if str(_GEMM_FLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMM_FLOPS_DIR))
from benchmark_modes import MODE_COLORS, MODE_DISPLAY, MODE_LINESTYLES, MODE_ORDER, add_shape_column, normalize_modes

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

FIDELITY_MARKERS = {
    "HiFi4": "s",
    "HiFi3": "D",
    "HiFi2": "o",
    "LoFi": "^",
}

FIDELITY_ORDER = ["HiFi4", "HiFi3", "HiFi2", "LoFi"]


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
    df["matrix_elements"] = df["m"] * df["k"] * df["n"]
    if df["use_trace"].dtype == object:
        df["use_trace"] = df["use_trace"].astype(str).str.lower() == "true"
    df = add_shape_column(df)
    return normalize_modes(df)


def _get_line_points(line_data, util_col_name, group_col, sort_col):
    best_rows = []
    line_data = line_data.dropna(subset=[util_col_name])
    for _, group in line_data.groupby(group_col, sort=True):
        best_rows.append(group.loc[group[util_col_name].idxmax()])
    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows).sort_values(sort_col)


def _plot_dtype_grid(df, dtype_short, device_prefix, device_label):
    if df.empty:
        print(f"    No data for {dtype_short} — skipping.")
        return

    host_col = _find_util_col(df, "Host")
    device_col = _find_util_col(df, "Device")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True, sharex=True)

    panel_defs = [
        (0, 0, "Host", False, host_col),
        (0, 1, "Host", True, host_col),
        (1, 0, "Device", False, device_col),
        (1, 1, "Device", True, device_col),
    ]

    all_fidelities = [f for f in FIDELITY_ORDER if f in df["fidelity"].unique()]
    all_modes = [mode for mode in MODE_ORDER if mode in df["mode"].unique()]

    for row, col_idx, util_kind, use_trace, util_col_name in panel_defs:
        ax = axes[row][col_idx]
        subset = df[df["use_trace"] == use_trace]

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
            ax.grid(True, alpha=0.3, linestyle="--")
            continue

        for fidelity in all_fidelities:
            marker = FIDELITY_MARKERS.get(fidelity, "x")
            for mode in all_modes:
                color = MODE_COLORS[mode]
                linestyle = MODE_LINESTYLES[mode]
                line_data = subset[(subset["mode"] == mode) & (subset["fidelity"] == fidelity)]
                if line_data.empty:
                    continue
                line_data = _get_line_points(line_data, util_col_name, "shape", "matrix_elements")
                if line_data.empty:
                    continue
                ax.plot(
                    line_data["matrix_elements"],
                    line_data[util_col_name],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    markersize=7,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=1,
                    linewidth=2,
                    alpha=0.85,
                )

        trace_label = "With Trace" if use_trace else "Without Trace"
        ax.set_title(f"{util_kind} | {trace_label}", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(0, 110)

    axes[1][0].set_xlabel("Total Matrix Elements (M × K × N)", fontsize=11, fontweight="bold")
    axes[1][1].set_xlabel("Total Matrix Elements (M × K × N)", fontsize=11, fontweight="bold")
    axes[0][0].set_ylabel("Utilization (%)", fontsize=11, fontweight="bold")
    axes[1][0].set_ylabel("Utilization (%)", fontsize=11, fontweight="bold")

    dtype_display = DTYPE_DISPLAY.get(dtype_short, dtype_short)
    fig.suptitle(
        f"Matmul Utilization — {device_label} — {dtype_display}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    legend_elements = [Line2D([0], [0], color="none", marker="none", label="Mode")]
    for mode in all_modes:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=MODE_COLORS[mode],
                linewidth=2.5,
                linestyle=MODE_LINESTYLES[mode],
                marker="none",
                label=MODE_DISPLAY.get(mode, mode),
            )
        )
    legend_elements.append(Line2D([0], [0], color="none", marker="none", label=""))
    legend_elements.append(Line2D([0], [0], color="none", marker="none", label="Fidelity"))
    for fid in all_fidelities:
        marker = FIDELITY_MARKERS.get(fid, "x")
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=0,
                marker=marker,
                markersize=9,
                markerfacecolor="gray",
                markeredgecolor="gray",
                label=fid,
            )
        )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        fontsize=9,
        framealpha=0.95,
        edgecolor="black",
        handlelength=3.5,
        bbox_to_anchor=(0.5, 0.01),
    )

    out_path = IMG_DIR / f"{device_prefix}-util-{dtype_short}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {out_path}")
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
            _plot_dtype_grid(df_dtype, dtype_short, device_prefix, device_label)


if __name__ == "__main__":
    main()
