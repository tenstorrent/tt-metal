#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate per-dtype 2x2 utilization grid plots from a single benchmark CSV.

One PNG per (device, dtype) combination.  Each PNG has 4 subplots:
  {host, device} x {no-trace, trace}.

Encoding:
  Line color -> mode:
                 DRAM (tuned_2d_dram) = blue
                 L1 (tuned_2d_l1) = red
                 OOB = black
  Line style -> tuned modes use solid (DRAM) / dashed (L1); OOB is solid
  Marker     -> math fidelity (fixed across all plots):
                 HiFi4=square, HiFi3=diamond, HiFi2=circle, LoFi=triangle

Reads a single CSV from tech_reports/GEMM_FLOPS/data/:
  {bh,wh}.csv  -- from test_matmul_2d_host_perf (unified benchmark)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DATA_DIR = Path("tech_reports/GEMM_FLOPS/data")
IMG_DIR = Path("tech_reports/GEMM_FLOPS/images")

DEVICE_MAP = {
    "bh": "P150 (Blackhole)",
    "wh": "N150 (Wormhole)",
}

BASE_SHAPE_COLUMNS = ["base_m", "base_k", "base_n"]

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

MODE_COLORS = {
    "tuned_2d_dram": "#1f77b4",
    "tuned_2d_l1": "#d62728",
    "oob": "black",
}

MODE_LINESTYLES = {
    "tuned_2d_dram": "-",
    "tuned_2d_l1": "--",
    "oob": "-",
}

MODE_DISPLAY = {
    "tuned_2d_dram": "DRAM (tuned)",
    "tuned_2d_l1": "L1 (tuned)",
    "oob": "OOB (auto)",
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


def _parse_grid_size(raw):
    cleaned = str(raw).strip("() ")
    grid_x, grid_y = [int(x.strip()) for x in cleaned.split(",")]
    return grid_x, grid_y


def _add_base_shape_columns(df):
    """Ensure base_m/base_k/base_n exist while preserving full scaled m/k/n."""
    if not all(col in df.columns for col in BASE_SHAPE_COLUMNS):
        grid_dims = df["grid_size"].apply(_parse_grid_size)
        df["base_m"] = [m // grid_y for m, (_, grid_y) in zip(df["m"], grid_dims)]
        df["base_k"] = [k // grid_x for k, (grid_x, _) in zip(df["k"], grid_dims)]
        df["base_n"] = [n // grid_x for n, (grid_x, _) in zip(df["n"], grid_dims)]
    df["base_shape"] = list(zip(df["base_m"], df["base_k"], df["base_n"]))
    return df


def _parse_dtype(raw):
    full = str(raw).split(".")[-1]
    return DTYPE_MAP.get(full, full.lower())


def _parse_fidelity(raw):
    return str(raw).split(".")[-1]


def _find_util_col(df, kind, grid_keyword="user selected grid"):
    """Find a utilization column. kind is 'Host' or 'Device'."""
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
    df = _add_base_shape_columns(df)
    return df


def _get_best_line_by_base_shape(line_data, util_col_name):
    best_rows = []
    line_data = line_data.dropna(subset=[util_col_name])
    for _, group in line_data.groupby("base_shape", sort=True):
        best_rows.append(group.loc[group[util_col_name].idxmax()])
    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows).sort_values("matrix_elements")


def _plot_dtype_grid(df_all, dtype_short, device_prefix, device_label):
    """Generate one 2x2 PNG for a given device and dtype."""
    df = df_all[df_all["dtype_short"] == dtype_short].copy()
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
    all_modes = [m for m in MODE_COLORS if m in df["mode"].unique()]

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
                line_data = _get_best_line_by_base_shape(line_data, util_col_name)
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

    axes[1][0].set_xlabel("Total Matrix Elements (M x K x N)", fontsize=11, fontweight="bold")
    axes[1][1].set_xlabel("Total Matrix Elements (M x K x N)", fontsize=11, fontweight="bold")
    axes[0][0].set_ylabel("Utilization (%)", fontsize=11, fontweight="bold")
    axes[1][0].set_ylabel("Utilization (%)", fontsize=11, fontweight="bold")

    dtype_display = DTYPE_DISPLAY.get(dtype_short, dtype_short)
    fig.suptitle(
        f"Matmul Utilization — {device_label} — {dtype_display}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Legend
    legend_elements = []

    legend_elements.append(Line2D([0], [0], color="none", marker="none", label="Mode"))
    for mode in all_modes:
        color = MODE_COLORS[mode]
        linestyle = MODE_LINESTYLES[mode]
        display = MODE_DISPLAY.get(mode, mode)
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, linestyle=linestyle, marker="none", label=display)
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

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = IMG_DIR / f"{device_prefix}-util-{dtype_short}.png"
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
        all_dtypes = sorted(df["dtype_short"].unique())
        for dtype_short in all_dtypes:
            _plot_dtype_grid(df, dtype_short, device_prefix, device_label)


if __name__ == "__main__":
    main()
