#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
3D M×N utilization surface plots — one PNG per (device, dtype, mode, fidelity).

Each PNG has a 2×2 grid ({host, device} × {no-trace, trace}).  Within each panel,
one colored surface is drawn per unique K value (utilization on Z).

Output layout (under images/):
  mn-3d/{device}-util-{dtype}-{mode}.png

Reads tech_reports/GEMM_FLOPS/data/{bh,wh}.csv
"""

from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

_GEMM_FLOPS_DIR = Path(__file__).resolve().parent
if str(_GEMM_FLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMM_FLOPS_DIR))
from benchmark_modes import MODE_DISPLAY, add_shape_column, normalize_modes

DATA_DIR = _GEMM_FLOPS_DIR / "data"
IMG_DIR = _GEMM_FLOPS_DIR / "images"
OUT_SUBDIR = "mn-3d"

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


def _surface_grid(slice_df, util_col, m_vals, n_vals):
    """Build (M, N, Z) mesh arrays for plot_surface; Z is NaN where no data."""
    if slice_df.empty:
        z = np.full((len(m_vals), len(n_vals)), np.nan)
        m_grid, n_grid = np.meshgrid(m_vals, n_vals, indexing="ij")
        return m_grid, n_grid, z

    best = slice_df.groupby(["m", "n"], sort=True)[util_col].max().reset_index()
    lookup = {((row["m"], row["n"])): row[util_col] for _, row in best.iterrows()}
    z = np.array([[lookup.get((m, n), np.nan) for n in n_vals] for m in m_vals], dtype=float)
    m_grid, n_grid = np.meshgrid(m_vals, n_vals, indexing="ij")
    return m_grid, n_grid, z


def _log_dim_ticks(dim_vals):
    """Return log10 tick positions and human-readable labels for a dimension axis."""
    log_ticks = np.log10(dim_vals)
    labels = [str(int(v)) if float(v).is_integer() else str(v) for v in dim_vals]
    return log_ticks, labels


def _mn_plane_coords_cw90(m_grid, n_grid, m_vals, n_vals):
    """Rotate the M×N plane 90° clockwise around Z (in log space).

    Maps (log M, log N) -> (log N, log M reflected) so the small-M/small-N
    corner sits at the deep/back side of the default 3D view.
    """
    log_m = np.log10(m_grid)
    log_n = np.log10(n_grid)
    log_m_min = np.log10(min(m_vals))
    log_m_max = np.log10(max(m_vals))
    x_plot = log_n
    y_plot = log_m_max + log_m_min - log_m
    return x_plot, y_plot


def _configure_log_mn_axes_rotated_cw90(ax, m_vals, n_vals):
    log_m_min = np.log10(min(m_vals))
    log_m_max = np.log10(max(m_vals))
    log_n_ticks, n_labels = _log_dim_ticks(n_vals)
    m_tick_pos = [log_m_max + log_m_min - np.log10(m) for m in m_vals]
    m_labels = [str(int(m)) if float(m).is_integer() else str(m) for m in m_vals]
    ax.set_xticks(log_n_ticks)
    ax.set_xticklabels(n_labels)
    ax.set_yticks(m_tick_pos)
    ax.set_yticklabels(m_labels)
    ax.set_xlabel("N (columns, log scale)", fontsize=10, labelpad=8)
    ax.set_ylabel("M (rows, log scale)", fontsize=10, labelpad=8)



def _darken_color(color, factor=0.55):
    """Return a darker RGB tuple for mesh edge lines."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _colors_for_k_values(k_values):
    cmap = plt.colormaps["turbo"]
    return {k: cmap(i / max(len(k_values) - 1, 1)) for i, k in enumerate(k_values)}


def _plot_mn_surfaces(
    df,
    dtype_short,
    device_prefix,
    device_label,
    mode,
    fidelity,
):
    k_values = sorted(df["k"].unique())
    m_vals = sorted(df["m"].unique())
    n_vals = sorted(df["n"].unique())
    if len(k_values) < 1 or len(m_vals) < 2 or len(n_vals) < 2:
        return

    host_col = _find_util_col(df, "Host")
    device_col = _find_util_col(df, "Device")
    k_colors = _colors_for_k_values(k_values)

    fig = plt.figure(figsize=(22, 16))
    panel_defs = [
        (1, "Host", False, host_col),
        (2, "Host", True, host_col),
        (3, "Device", False, device_col),
        (4, "Device", True, device_col),
    ]

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=k_colors[k],
            linewidth=2.5,
            marker="s",
            markersize=6,
            markerfacecolor=k_colors[k],
            markeredgecolor=k_colors[k],
            label=f"K={k}",
        )
        for k in k_values
    ]

    for subplot_idx, util_kind, use_trace, util_col_name in panel_defs:
        ax = fig.add_subplot(2, 2, subplot_idx, projection="3d")
        if util_col_name is None:
            ax.text2D(
                0.5,
                0.5,
                f"No {util_kind} data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            trace_label = "With Trace" if use_trace else "Without Trace"
            ax.set_title(f"{util_kind} | {trace_label}", fontsize=11, fontweight="bold", pad=12)
            continue

        subset = df[df["use_trace"] == use_trace]
        for k in k_values:
            k_df = subset[subset["k"] == k]
            m_grid, n_grid, z_grid = _surface_grid(k_df, util_col_name, m_vals, n_vals)
            if np.all(np.isnan(z_grid)):
                continue
            face_color = k_colors[k]
            edge_color = _darken_color(face_color)
            x_plot, y_plot = _mn_plane_coords_cw90(m_grid, n_grid, m_vals, n_vals)
            ax.plot_surface(
                x_plot,
                y_plot,
                z_grid,
                color=face_color,
                alpha=0.55,
                linewidth=0,
                antialiased=True,
                shade=True,
            )
            ax.plot_wireframe(
                x_plot,
                y_plot,
                z_grid,
                color=edge_color,
                linewidth=0.55,
                alpha=0.95,
                rstride=1,
                cstride=1,
            )

        trace_label = "With Trace" if use_trace else "Without Trace"
        ax.set_title(f"{util_kind} | {trace_label}", fontsize=11, fontweight="bold", pad=12)
        _configure_log_mn_axes_rotated_cw90(ax, m_vals, n_vals)
        ax.set_zlabel("Utilization (%)", fontsize=10, labelpad=8)
        ax.set_zlim(0, 110)
        ax.view_init(elev=28, azim=-58)
        ax.grid(True, alpha=0.25)

    mode_display = MODE_DISPLAY.get(mode, mode)
    dtype_display = DTYPE_DISPLAY.get(dtype_short, dtype_short)
    fig.suptitle(
        f"Matmul Utilization — M×N surfaces per K — {device_label} — {dtype_display}\n"
        f"{mode_display}, {fidelity}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        fontsize=9,
        framealpha=0.95,
        edgecolor="black",
        handlelength=2.5,
        bbox_to_anchor=(0.5, 0.01),
    )

    out_dir = IMG_DIR / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{device_prefix}-util-{dtype_short}-{mode}.png"

    fig.subplots_adjust(top=0.90, bottom=0.08, wspace=0.12, hspace=0.18)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {out_path} ({len(k_values)} K surfaces)")
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
                    _plot_mn_surfaces(
                        subset,
                        dtype_short,
                        device_prefix,
                        device_label,
                        mode,
                        fidelity,
                    )


if __name__ == "__main__":
    main()
