#!/usr/bin/env python3
"""
Generate per-dtype 2x2 utilization grid plots.

One PNG per (device, dtype) combination.  Each PNG has 4 subplots:
  {host, device} x {no-trace, trace}.

Encoding:
  Line color -> memory config:
                 DRAM=blue, L1=red, OOB=black
  Marker     -> math fidelity (fixed across all plots):
                 HiFi4=square, HiFi3=diamond, HiFi2=circle, LoFi=triangle

OOB lines are per-fidelity (one black line per fidelity level).

Reads CSVs from tech_reports/GEMM_FLOPS/data/:
  {bh,wh}-tuned.csv  -- from test_matmul_2d_host_perf
  {bh,wh}-oob.csv    -- from test_matmul_2d_host_perf_out_of_box
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

MEM_COLORS = {
    "DRAM": "#1f77b4",
    "L1": "#d62728",
    "OOB": "black",
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
    """Find a utilization column. kind is 'Host' or 'Device'."""
    prefix = f"{kind} based utilization"
    for col in df.columns:
        if col.startswith(prefix) and grid_keyword in col:
            if df[col].notna().any():
                return col
    return None


def _load_tuned(path):
    df = _read_csv(path)
    if df.empty:
        return df
    df["dtype_short"] = df["dtype"].apply(_parse_dtype)
    df["fidelity"] = df["math_fidelity"].apply(_parse_fidelity)
    df["mem"] = df["in0_sharded"].apply(lambda x: "L1" if x else "DRAM")
    df["matrix_elements"] = df["m"] * df["k"] * df["n"]
    df["source"] = "tuned"
    return df


def _load_oob(path):
    df = _read_csv(path)
    if df.empty:
        return df
    df["dtype_short"] = df["dtype"].apply(_parse_dtype)
    df["fidelity"] = df["math_fidelity"].apply(_parse_fidelity)
    df["mem"] = "OOB"
    df["matrix_elements"] = df["m"] * df["k"] * df["n"]
    df["source"] = "oob"
    return df


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

            for mem in ["DRAM", "L1"]:
                color = MEM_COLORS[mem]
                linestyle = "-" if mem == "DRAM" else "--"
                line_data = subset[
                    (subset["mem"] == mem) & (subset["fidelity"] == fidelity) & (subset["source"] == "tuned")
                ]
                if line_data.empty:
                    continue
                line_data = line_data.sort_values("matrix_elements")
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

            oob_data = subset[(subset["source"] == "oob") & (subset["fidelity"] == fidelity)]
            if not oob_data.empty:
                oob_color = MEM_COLORS["OOB"]
                oob_data = oob_data.sort_values("matrix_elements")
                ax.plot(
                    oob_data["matrix_elements"],
                    oob_data[util_col_name],
                    marker=marker,
                    linestyle="-",
                    color=oob_color,
                    markersize=7,
                    markerfacecolor=oob_color,
                    markeredgecolor=oob_color,
                    markeredgewidth=1,
                    linewidth=2,
                    alpha=0.9,
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

    legend_elements = []

    legend_elements.append(Line2D([0], [0], color="none", marker="none", label="Memory"))
    legend_elements.append(
        Line2D(
            [0], [0], color=MEM_COLORS["DRAM"], linewidth=2.5, linestyle="-", marker="none", label="DRAM (interleaved)"
        )
    )
    legend_elements.append(
        Line2D([0], [0], color=MEM_COLORS["L1"], linewidth=2.5, linestyle="--", marker="none", label="L1 (sharded)")
    )
    legend_elements.append(
        Line2D([0], [0], color=MEM_COLORS["OOB"], linewidth=2.5, linestyle="-", marker="none", label="OOB (auto)")
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
        tuned_path = DATA_DIR / f"{device_prefix}-tuned.csv"
        oob_path = DATA_DIR / f"{device_prefix}-oob.csv"

        df_tuned = _load_tuned(tuned_path)
        df_oob = _load_oob(oob_path)

        if df_tuned.empty and df_oob.empty:
            print(f"No data for {device_label} — skipping.")
            continue

        df_all = pd.concat([df_tuned, df_oob], ignore_index=True)
        print(f"{device_label}:")

        all_dtypes = sorted(df_all["dtype_short"].unique())
        for dtype_short in all_dtypes:
            _plot_dtype_grid(df_all, dtype_short, device_prefix, device_label)


if __name__ == "__main__":
    main()
