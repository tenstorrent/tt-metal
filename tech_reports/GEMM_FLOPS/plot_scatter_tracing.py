# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Trace comparison plot: traced vs non-traced TFLOPs per device.

Usage:
1. Run the benchmark via run_bench.sh on both devices
2. CSVs are placed in tech_reports/GEMM_FLOPS/data/{wh,bh}.csv
3. Run this script from the tt-metal root directory
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = Path("tech_reports/GEMM_FLOPS/data")
IMG_DIR = Path("tech_reports/GEMM_FLOPS/images")

DEVICE_FILES = {
    "n150": DATA_DIR / "wh.csv",
    "p150": DATA_DIR / "bh.csv",
}

DEVICE_LABELS = {
    "n150": "N150 (Wormhole)",
    "p150": "P150 (Blackhole)",
}

BASE_SHAPE_COLUMNS = ["base_m", "base_k", "base_n"]

dtype_configs = [
    ("BFLOAT4_B_LoFi", "BFLOAT4_B (LoFi)", "#2ca02c"),  # Green
    ("BFLOAT8_B_HiFi2", "BFLOAT8_B (HiFi2)", "#ff7f0e"),  # Orange
    ("BFLOAT16_HiFi4", "BFLOAT16 (HiFi4)", "#1f77b4"),  # Blue
]


def safe_read_csv(path):
    """Return the CSV as a DataFrame, or an empty DataFrame if the file is missing."""
    if path.exists():
        return pd.read_csv(path)
    print(f"WARNING: {path} not found — skipping that device.")
    return pd.DataFrame()


def parse_grid_size(raw):
    cleaned = str(raw).strip("() ")
    grid_x, grid_y = [int(x.strip()) for x in cleaned.split(",")]
    return grid_x, grid_y


def add_base_shape_columns(df):
    """Ensure base_m/base_k/base_n exist while preserving full scaled m/k/n."""
    if not all(col in df.columns for col in BASE_SHAPE_COLUMNS):
        grid_dims = df["grid_size"].apply(parse_grid_size)
        df["base_m"] = [m // grid_y for m, (_, grid_y) in zip(df["m"], grid_dims)]
        df["base_k"] = [k // grid_x for k, (grid_x, _) in zip(df["k"], grid_dims)]
        df["base_n"] = [n // grid_x for n, (grid_x, _) in zip(df["n"], grid_dims)]
    df["base_shape"] = list(zip(df["base_m"], df["base_k"], df["base_n"]))
    return df


def load_and_prepare(path, source):
    """Load CSV and prepare derived columns."""
    df = safe_read_csv(path)
    if df.empty:
        return df
    df["source"] = source
    # Use best performance across all tuned modes (exclude OOB for peak comparison)
    if "mode" in df.columns:
        df = df[df["mode"] != "oob"].copy()
    if "TFLOPs (avg)" in df.columns:
        df.rename(columns={"TFLOPs (avg)": "tflops"}, inplace=True)
    df["tflops"] = pd.to_numeric(df["tflops"], errors="coerce")
    df["dtype_fidelity"] = (
        df["dtype"].astype(str).str.replace("DataType.", "")
        + "_"
        + df["math_fidelity"].astype(str).str.replace("MathFidelity.", "")
    )
    df["matrix_elements"] = df["m"] * df["k"] * df["n"]
    if df["use_trace"].dtype == object:
        df["use_trace"] = df["use_trace"].astype(str).str.lower() == "true"
    df = add_base_shape_columns(df)
    return df


def get_best_trace_pairs(df_slice):
    """Return plotted series and ratio points keyed by exact base matrix shape."""
    traced_perf = []
    nontraced_perf = []
    ratio_points = []

    for _, shape_group in df_slice.groupby("base_shape", sort=True):
        traced_group = shape_group[shape_group["use_trace"]]
        nontraced_group = shape_group[~shape_group["use_trace"]]

        best_traced_tflops = None
        best_nontraced_tflops = None
        matrix_elements = shape_group["matrix_elements"].iloc[0]
        if not traced_group.empty:
            best_traced_row = traced_group.loc[traced_group["tflops"].idxmax()]
            best_traced_tflops = best_traced_row["tflops"]
            traced_perf.append((best_traced_row["matrix_elements"], best_traced_tflops))
        if not nontraced_group.empty:
            best_nontraced_row = nontraced_group.loc[nontraced_group["tflops"].idxmax()]
            best_nontraced_tflops = best_nontraced_row["tflops"]
            nontraced_perf.append((best_nontraced_row["matrix_elements"], best_nontraced_tflops))

        if (
            best_traced_tflops is not None
            and best_nontraced_tflops is not None
            and pd.notna(best_traced_tflops)
            and pd.notna(best_nontraced_tflops)
            and best_nontraced_tflops > 0
        ):
            ratio_points.append((matrix_elements, best_traced_tflops, best_nontraced_tflops))

    traced_perf.sort()
    nontraced_perf.sort()
    ratio_points.sort()
    return traced_perf, nontraced_perf, ratio_points


frames = []
for source, path in DEVICE_FILES.items():
    loaded = load_and_prepare(path, source)
    if not loaded.empty:
        frames.append(loaded)

if not frames:
    print("ERROR: No data available for any device. Exiting.")
    raise SystemExit(1)

df = pd.concat(frames, ignore_index=True)

IMG_DIR.mkdir(parents=True, exist_ok=True)

available_sources = [s for s in DEVICE_FILES if s in df["source"].values]
for source in available_sources:
    fig, ax = plt.subplots(figsize=(16, 10))
    device_data = df[df["source"] == source].copy()

    for dtype_fidelity, dtype_label, color in dtype_configs:
        df_slice = device_data[device_data["dtype_fidelity"] == dtype_fidelity]
        traced_perf, nontraced_perf, ratio_points = get_best_trace_pairs(df_slice)

        if not traced_perf or not nontraced_perf:
            continue

        traced_x, traced_y = zip(*traced_perf)
        nontraced_x, nontraced_y = zip(*nontraced_perf)

        ax.plot(
            traced_x, traced_y, color=color, linestyle="-", linewidth=2.5, label=f"{dtype_label} (Traced)", zorder=3
        )
        ax.scatter(traced_x, traced_y, color=color, marker="^", s=120, edgecolors="black", linewidths=1.2, zorder=4)

        ax.plot(
            nontraced_x,
            nontraced_y,
            color=color,
            linestyle="--",
            linewidth=2.5,
            label=f"{dtype_label} (Non-traced)",
            zorder=3,
        )
        ax.scatter(
            nontraced_x,
            nontraced_y,
            color=color,
            marker="v",
            s=120,
            facecolors="none",
            edgecolors=color,
            linewidths=2,
            zorder=4,
        )

        for x, y_trace, y_nontrace in ratio_points:
            ratio = y_trace / y_nontrace
            y_pos = max(y_trace, y_nontrace)
            ax.annotate(
                f"{ratio:.2f}×",
                (x, y_pos),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85, linewidth=1),
                zorder=5,
            )

    ax.set_xscale("log")
    ax.text(
        0.5,
        -0.08,
        "Total Matrix Elements (M × K × N)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        -0.12,
        "[(M,K) = input matrix size, (K,N) = weight matrix size]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    ax.set_ylabel("Performance (TFLOPs)", fontsize=14, fontweight="bold", labelpad=10)

    device_name = DEVICE_LABELS.get(source, source)
    fig.suptitle(f"Performance Comparison: {device_name}", fontsize=18, fontweight="bold", y=0.98)
    ax.set_title(
        "Traced vs Non-Traced Execution Performance (All Matrix Sizes)", fontsize=14, pad=10, fontweight="bold"
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)

    legend_elements = []

    # Dtype section header
    legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Dtype\ (Math\ Fidelity)}$"))

    # Add each dtype with its color
    for dtype_fidelity, dtype_label, color in dtype_configs:
        legend_elements.append(Line2D([0], [0], color=color, linewidth=4, label=dtype_label))

    # Spacer
    legend_elements.append(Line2D([0], [0], color="none", label=""))

    # Execution type section header
    legend_elements.append(Line2D([0], [0], color="none", label=r"$\mathbf{Execution\ Type}$"))

    # Traced (solid line, filled triangles)
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=3,
            linestyle="-",
            marker="^",
            markersize=10,
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=1.5,
            label="Traced",
        )
    )

    # Non-traced (dashed line, hollow triangles)
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=3,
            linestyle="--",
            marker="v",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="gray",
            markeredgewidth=2.5,
            label="Non-traced",
        )
    )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=12,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
        shadow=True,
        handlelength=3.5,
    )

    plt.tight_layout()
    plt.savefig(IMG_DIR / f"trace_comparison_{source}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: trace_comparison_{source}.png")

print("\nTracing comparison plots complete!")
