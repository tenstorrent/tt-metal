# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for finding best configurations, tracing comparisons, and OOB comparisons
from GEMM sweep data.
"""

import pandas as pd
import numpy as np


def load_sweep_data(
    n150_path="tech_reports/GEMM_FLOPS/n150-sweep.csv",
    p150_path="tech_reports/GEMM_FLOPS/p150-sweep.csv",
    combined_path=None,
):
    """Load and standardize sweep data from both devices or from a combined file."""

    if combined_path:
        # Load from combined file (like matmul_2d_host_perf_sweep_all.csv)
        df = pd.read_csv(combined_path)

        # Determine source from grid_size column if available
        if "grid_size" in df.columns:
            df["source"] = df["grid_size"].apply(lambda x: "n150" if "8x8" in str(x) else "p150")
        else:
            # Fallback: no source info available
            df["source"] = "unknown"
    else:
        # Load from separate files
        df_n150 = pd.read_csv(n150_path)
        df_p150 = pd.read_csv(p150_path)

        # Standardize column names
        if "best_tflops" in df_p150.columns:
            df_p150.rename(columns={"best_tflops": "tflops"}, inplace=True)

        # Add source column
        df_n150["source"] = "n150"
        df_p150["source"] = "p150"

        # Combine dataframes
        df = pd.concat([df_n150, df_p150], ignore_index=True)

    # Create dtype_fidelity column
    df["dtype_short"] = df["dtype"].str.replace("DataType.", "", regex=False)
    df["math_fidelity_short"] = df["math_fidelity"].str.replace("MathFidelity.", "", regex=False)
    df["dtype_fidelity"] = df["dtype_short"] + "_" + df["math_fidelity_short"]

    # Calculate matrix elements
    df["matrix_elements"] = df["m"] * df["k"] * df["n"]

    # Add utilization columns - use host-based utilization as device-based has calculation errors
    df["utilization"] = np.nan
    n150_util_col = "Host based utilization[%] (vs full available grid 8x8)"
    p150_util_col = "Host based utilization[%] (vs full available grid 13x10)"

    if n150_util_col in df.columns:
        df.loc[df["source"] == "n150", "utilization"] = df[n150_util_col]
    if p150_util_col in df.columns:
        df.loc[df["source"] == "p150", "utilization"] = df[p150_util_col]

    # Add aspect ratio pattern column if aspect ratio columns exist
    if all(col in df.columns for col in ["aspect_ratio_m", "aspect_ratio_k", "aspect_ratio_n"]):
        df["aspect_ratio_pattern"] = df.apply(
            lambda row: f"{int(row['aspect_ratio_m'])}:{int(row['aspect_ratio_k'])}:{int(row['aspect_ratio_n'])}",
            axis=1,
        )

    return df


def get_best_config_with_storage_precedence(group, metric="tflops"):
    """
    Get best config with smart storage precedence: prefer L1 only if it performs significantly better.

    Args:
        group: DataFrame group to search within.
        metric: The column to maximize ('tflops' or 'utilization').
    """
    if group.empty:
        return pd.Series(dtype="object")

    # Get best L1 and DRAM configs
    l1_configs = group[(group["in0_sharded"] == True) | (group["out_sharded"] == True)]  # L1 storage via sharding

    dram_configs = group[(group["in0_sharded"] == False) & (group["out_sharded"] == False)]  # DRAM storage

    best_l1 = l1_configs.loc[l1_configs[metric].idxmax()] if not l1_configs.empty else None
    best_dram = dram_configs.loc[dram_configs[metric].idxmax()] if not dram_configs.empty else None

    # If both exist, choose based on performance with slight L1 preference
    if best_l1 is not None and best_dram is not None:
        l1_perf = best_l1[metric]
        dram_perf = best_dram[metric]

        # Prefer L1 only if it's at least 95% as good as DRAM (small tolerance for L1 preference)
        if l1_perf >= dram_perf * 0.95:
            return best_l1
        else:
            return best_dram

    # If only one type exists, use that
    if best_l1 is not None:
        return best_l1
    if best_dram is not None:
        return best_dram

    # Fallback to overall best
    return group.loc[group[metric].idxmax()]


def get_default_oob_config(group):
    """
    Get the default OOB config (DRAM, all params=1, no tracing) for a tensor shape.
    If exact default not found, find the most default-like configuration.
    """
    if group.empty:
        return pd.Series(dtype="object")

    # Try exact default first
    default_config = group[
        (group["in0_sharded"] == False)
        & (group["out_sharded"] == False)
        & (group["in0_block_w_div"] == 1)
        & (group["num_out_blocks_h"] == 1)
        & (group["num_out_blocks_w"] == 1)
        & (group["use_trace"] == False)
    ]

    if not default_config.empty:
        return default_config.iloc[0]

    # If exact default not found, find most default-like: DRAM + no tracing + minimal block params
    fallback_configs = group[
        (group["in0_sharded"] == False) & (group["out_sharded"] == False) & (group["use_trace"] == False)
    ]

    if not fallback_configs.empty:
        # Find config closest to default (1,1,1) - this should be the true OOB baseline
        fallback_configs = fallback_configs.copy()
        fallback_configs["default_distance"] = (
            (fallback_configs["in0_block_w_div"] - 1).abs()
            + (fallback_configs["num_out_blocks_h"] - 1).abs()
            + (fallback_configs["num_out_blocks_w"] - 1).abs()
        )

        # Find the most default-like config(s) and pick the first one
        min_distance = fallback_configs["default_distance"].min()
        closest_configs = fallback_configs[fallback_configs["default_distance"] == min_distance]
        return closest_configs.iloc[0]  # Take first match for deterministic behavior

    return pd.Series(dtype="object")


def get_best_config_vs_default(group, default_perf):
    """
    Get best config, but use default if it performs better.
    """
    if group.empty:
        return pd.Series(dtype="object")

    best_config = group.loc[group["tflops"].idxmax()]

    if default_perf >= best_config["tflops"]:
        return get_default_oob_config(group)

    return best_config


def get_bar_chart_data(df, precision_fidelity_pairs, matrix_size_mapping):
    """Gets and processes data specifically for the bar chart plot."""

    # Make sure tflops is numeric
    df["tflops"] = pd.to_numeric(df["tflops"], errors="coerce")

    # Filter for near-square matrices (k=n)
    df_filtered = df[df["k"] == df["n"]].copy()

    if df_filtered.empty:
        # No k=n matrices found, check for square base matrices (m=k=n)
        df_filtered = df[(df["k"] == df["n"]) & (df["m"] == df["k"])].copy()

        if df_filtered.empty:
            # No square matrices found, use all data
            df_filtered = df.copy()

    df_filtered["matrix_size"] = df_filtered.apply(lambda row: f"{row['m']}x{row['k']}x{row['n']}", axis=1)

    # Find best performance for each group
    best_rows = []
    for (matrix_size, dtype_fidelity, source), group in df_filtered.groupby(
        ["matrix_size", "dtype_fidelity", "source"]
    ):
        best_entry = get_best_config_with_storage_precedence(group)
        if not best_entry.empty:
            best_rows.append(best_entry)
    df_best = pd.DataFrame(best_rows)

    if df_best.empty:
        return {}

    # Create combined labels for the x-axis
    df_best["total_elements"] = df_best["m"] * df_best["k"] * df_best["n"]
    matrix_sizes_sorted = df_best.sort_values("total_elements")["matrix_size"].unique()

    # --- Grouping and data extraction ---
    grouped_sizes = {}
    for n150_size_str in matrix_sizes_sorted:
        if "n150" not in df_best[df_best["matrix_size"] == n150_size_str]["source"].unique():
            continue
        n150_dim = n150_size_str.split("x")[0]
        if n150_dim in matrix_size_mapping:
            p150_base_dim = matrix_size_mapping[n150_dim]
            grouped_sizes[f"{n150_dim}/{p150_base_dim}"] = (n150_size_str, p150_base_dim)

    # --- Data Collection ---
    chart_data = {}
    for combined_label, (n150_size, p150_base_dim) in grouped_sizes.items():
        chart_data[combined_label] = {"n150": {}, "p150": {}}
        group_n150 = df_best[df_best["matrix_size"] == n150_size]

        # Find best p150 match for each dtype
        p150_all_for_m = df_best[(df_best["source"] == "p150") & (df_best["m"] == int(p150_base_dim))]

        for dtype, fidelity in precision_fidelity_pairs:
            # N150 data
            n150_entry = group_n150[
                (group_n150["dtype_short"] == dtype) & (group_n150["math_fidelity_short"] == fidelity)
            ]
            if not n150_entry.empty:
                chart_data[combined_label]["n150"][dtype] = n150_entry.iloc[0]["tflops"]

            # P150 data
            p150_candidates = p150_all_for_m[
                (p150_all_for_m["dtype_short"] == dtype) & (p150_all_for_m["math_fidelity_short"] == fidelity)
            ]
            if not p150_candidates.empty:
                best_p150_row = p150_candidates.loc[p150_candidates["tflops"].idxmax()]
                chart_data[combined_label]["p150"][dtype] = best_p150_row["tflops"]

    return chart_data
