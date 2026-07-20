# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Aspect ratio comparison: TFLOPs by matrix shape aspect ratio per device.

Usage:
1. Run the benchmark via run_bench.sh on both devices
2. CSVs are placed in tech_reports/GEMM_FLOPS/data/{wh,bh}.csv
3. Run this script from the tt-metal root directory
"""

from pathlib import Path
import sys
from functools import reduce
from math import gcd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_GEMM_FLOPS_DIR = Path(__file__).resolve().parent
if str(_GEMM_FLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMM_FLOPS_DIR))
from benchmark_modes import add_shape_column, normalize_modes

DATA_DIR = _GEMM_FLOPS_DIR / "data"
IMG_DIR = _GEMM_FLOPS_DIR / "images"

DEVICE_FILES = {
    "N150": DATA_DIR / "wh.csv",
    "P150": DATA_DIR / "bh.csv",
}

DEVICE_LABELS = {
    "N150": "N150 (Wormhole)",
    "P150": "P150 (Blackhole)",
}


def safe_read_csv(path):
    """Return the CSV as a DataFrame, or an empty DataFrame if the file is missing."""
    if path.exists():
        return pd.read_csv(path)
    print(f"WARNING: {path} not found — skipping that device.")
    return pd.DataFrame()


def gcd_of_three(a, b, c):
    """Calculate GCD of three numbers"""
    return reduce(gcd, [a, b, c])


def calculate_aspect_ratio(m, k, n):
    """Calculate aspect ratio from matrix dimensions."""
    divisor = gcd_of_three(m, k, n)
    ratio_m = m // divisor
    ratio_k = k // divisor
    ratio_n = n // divisor

    return f"{ratio_m}:{ratio_k}:{ratio_n}"


def load_and_prepare(path, source):
    """Load CSV and prepare derived columns."""
    df = safe_read_csv(path)
    if df.empty:
        return df
    df["source"] = source
    # Use best performance across all tuned modes
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
    df = add_shape_column(df)
    df["aspect_ratio"] = df.apply(lambda row: calculate_aspect_ratio(row["m"], row["k"], row["n"]), axis=1)
    return normalize_modes(df)


# Load data
frames = []
for source, path in DEVICE_FILES.items():
    loaded = load_and_prepare(path, source)
    if not loaded.empty:
        frames.append(loaded)

if not frames:
    print("ERROR: No data available for any device. Exiting.")
    raise SystemExit(1)

df = pd.concat(frames, ignore_index=True)

# Define dtype-fidelity pairs and aspect ratios (colors match scatter/bar plots)
dtype_configs = [
    ("BFLOAT4_B_LoFi", "BFLOAT4_B (LoFi)", "#2ca02c"),  # Green
    ("BFLOAT8_B_HiFi2", "BFLOAT8_B (HiFi2)", "#ff7f0e"),  # Orange
    ("BFLOAT16_HiFi4", "BFLOAT16 (HiFi4)", "#1f77b4"),  # Blue
]

aspect_ratios = ["1:1:1", "1:2:4"]
aspect_labels = {"1:1:1": "Square\n(1:1:1)", "1:2:4": "Rectangular\n(1:2:4)"}

# Create plot for each device that has data
IMG_DIR.mkdir(parents=True, exist_ok=True)
print("Calculating aspect ratios...")

available_sources = [s for s in DEVICE_FILES if s in df["source"].values]
for source in available_sources:
    device_data = df[df["source"] == source].copy()

    if device_data.empty:
        print(f"\nNo data for {source}")
        continue

    # Collect data: for each aspect ratio, get all 3 dtypes
    TOP_N_SIZES = 3
    summary_data = []

    for aspect_ratio in aspect_ratios:
        for dtype_fidelity, dtype_label, color in dtype_configs:
            filtered = device_data[
                (device_data["dtype_fidelity"] == dtype_fidelity) & (device_data["aspect_ratio"] == aspect_ratio)
            ]

            if filtered.empty:
                print(f"  No data for {source} {dtype_fidelity} with aspect {aspect_ratio}")
                summary_data.append(
                    {
                        "aspect_ratio": aspect_ratio,
                        "dtype_fidelity": dtype_fidelity,
                        "dtype_label": dtype_label,
                        "avg_tflops": 0,
                        "color": color,
                    }
                )
                continue

            # Group by matrix size and get best TFLOPs for each size
            size_tflops = []
            for _, group in filtered.groupby("shape"):
                total_elements = group["m"].iloc[0] * group["k"].iloc[0] * group["n"].iloc[0]
                best_tflops = group["tflops"].max()
                size_tflops.append((total_elements, best_tflops))

            # Sort by total elements and take TOP_N largest
            size_tflops.sort(key=lambda x: x[0], reverse=True)
            top_n_tflops = [tflops for _, tflops in size_tflops[:TOP_N_SIZES]]

            if not top_n_tflops:
                avg_tflops = 0
            else:
                avg_tflops = np.mean(top_n_tflops)

            summary_data.append(
                {
                    "aspect_ratio": aspect_ratio,
                    "dtype_fidelity": dtype_fidelity,
                    "dtype_label": dtype_label,
                    "avg_tflops": avg_tflops,
                    "color": color,
                }
            )

            print(
                f"  {source} {dtype_fidelity} ({aspect_ratio}): {avg_tflops:.1f} TFLOPs "
                f"(avg of top {len(top_n_tflops)} sizes out of {len(size_tflops)} total)"
            )

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    n_dtypes = len(dtype_configs)
    bar_width = 0.22
    group_gap = 0.5

    aspect_positions = []
    current_pos = 0

    for aspect_idx, aspect_ratio in enumerate(aspect_ratios):
        for dtype_idx, (dtype_fidelity, dtype_label, color) in enumerate(dtype_configs):
            data_point = next(
                (
                    d
                    for d in summary_data
                    if d["aspect_ratio"] == aspect_ratio and d["dtype_fidelity"] == dtype_fidelity
                ),
                None,
            )

            if data_point:
                value = data_point["avg_tflops"]
                bar_pos = current_pos + dtype_idx * bar_width

                ax.bar(
                    bar_pos,
                    value,
                    bar_width,
                    color=color,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=1.3,
                    label=dtype_label if aspect_idx == 0 else "",
                )

                if value > 0:
                    ax.text(bar_pos, value, f"{value:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        aspect_center = current_pos + (n_dtypes * bar_width) / 2 - bar_width / 2
        aspect_positions.append(aspect_center)

        current_pos += n_dtypes * bar_width + group_gap

    ax.set_ylabel("Average TFLOPs", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_xlabel(
        "Matrix Aspect Ratio (M:K:N)",
        fontsize=13,
        fontweight="bold",
        labelpad=10,
    )
    ax.text(
        0.5,
        -0.12,
        "(M=input rows, K=inner dim, N=output cols)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    device_name = DEVICE_LABELS.get(source, source)
    fig.suptitle(f"Performance Comparison: {device_name}", fontsize=18, fontweight="bold", y=0.98)
    ax.set_title("TFLOPs by Matrix Aspect Ratio and Data Type", fontsize=14, pad=10, fontweight="bold")

    ax.set_xticks(aspect_positions)
    ax.set_xticklabels([aspect_labels[r] for r in aspect_ratios], fontsize=12)

    ax.legend(
        fontsize=11,
        loc="upper right",
        framealpha=0.95,
        edgecolor="black",
        title="Data Type (Math Fidelity)",
        title_fontsize=12,
    )

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(IMG_DIR / f"aspect_ratio_by_dtype_{source.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: aspect_ratio_by_dtype_{source.lower()}.png")

print("\nAspect ratio by dtype comparison complete!")
