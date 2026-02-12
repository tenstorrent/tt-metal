#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Analyze parameter sweep results for experts matmul.

This script loads the JSON results from parameter sweep tests and provides
analysis and visualization of the performance characteristics.

Usage:
    python analyze_param_sweep.py [result_file.json]

If no file is specified, it will use the most recent result file.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger


def load_results(result_file: str) -> pd.DataFrame:
    """Load parameter sweep results from JSON file."""
    with open(result_file, "r") as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # Convert list columns to tuples for groupby operations
    if "gate_up_cores" in df.columns:
        df["gate_up_cores"] = df["gate_up_cores"].apply(tuple)
    if "down_cores" in df.columns:
        df["down_cores"] = df["down_cores"].apply(tuple)

    return df


def find_latest_result() -> str:
    """Find the most recent result file in param_sweep_results directory."""
    result_dir = Path("param_sweep_results")
    if not result_dir.exists():
        logger.error("param_sweep_results directory not found")
        sys.exit(1)

    json_files = list(result_dir.glob("*.json"))
    if not json_files:
        logger.error("No result files found in param_sweep_results/")
        sys.exit(1)

    # Sort by modification time
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    logger.info("=" * 80)
    logger.info("PARAMETER SWEEP ANALYSIS")
    logger.info("=" * 80)

    # Overall stats
    total_configs = len(df)
    passed_configs = df["passed"].sum()
    failed_configs = total_configs - passed_configs

    logger.info(f"\nTotal configurations tested: {total_configs}")
    logger.info(f"Passed: {passed_configs} ({100*passed_configs/total_configs:.1f}%)")
    logger.info(f"Failed: {failed_configs} ({100*failed_configs/total_configs:.1f}%)")

    # Performance stats for passed configs
    df_passed = df[df["passed"] == True]
    if len(df_passed) > 0:
        logger.info(f"\nPerformance Statistics (passed configs):")
        logger.info(f"  Best (min):     {df_passed['avg_time_us'].min():.2f} us")
        logger.info(f"  Worst (max):    {df_passed['avg_time_us'].max():.2f} us")
        logger.info(f"  Mean:           {df_passed['avg_time_us'].mean():.2f} us")
        logger.info(f"  Median:         {df_passed['avg_time_us'].median():.2f} us")
        logger.info(f"  Std Dev:        {df_passed['avg_time_us'].std():.2f} us")

        # PCC stats
        logger.info(f"\nAccuracy Statistics:")
        logger.info(f"  Min PCC:        {df_passed['pcc'].min():.6f}")
        logger.info(f"  Mean PCC:       {df_passed['pcc'].mean():.6f}")


def print_best_configs(df: pd.DataFrame, top_n: int = 10):
    """Print the best performing configurations."""
    df_passed = df[df["passed"] == True].copy()
    if len(df_passed) == 0:
        logger.warning("No passed configurations to display")
        return

    # Sort by performance
    df_sorted = df_passed.sort_values("avg_time_us")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"TOP {top_n} CONFIGURATIONS")
    logger.info(f"{'=' * 80}")
    logger.info(f"\n{'Rank':<6} {'Time (us)':<12} {'PCC':<10} {'Cores':<12} {'in0_blk':<8}")
    logger.info("-" * 80)

    for i, row in df_sorted.head(top_n).iterrows():
        rank = df_sorted.index.get_loc(i) + 1
        cores = str(row["cores"]) if "cores" in row else "N/A"
        logger.info(
            f"{rank:<6} {row['avg_time_us']:<12.2f} {row['pcc']:<10.6f} " f"{cores:<12} {row['in0_block_w']:<8}"
        )


def analyze_parameter_impact(df: pd.DataFrame):
    """Analyze the impact of individual parameters."""
    df_passed = df[df["passed"] == True]
    if len(df_passed) == 0:
        return

    logger.info(f"\n{'=' * 80}")
    logger.info("PARAMETER IMPACT ANALYSIS")
    logger.info(f"{'=' * 80}")

    # Analyze in0_block_w impact
    logger.info("\nImpact of in0_block_w:")
    grouped = df_passed.groupby("in0_block_w")["avg_time_us"].agg(["count", "mean", "min", "max"])
    logger.info(grouped.to_string())

    # Analyze core grid impact
    logger.info("\nImpact of gate_up_cores:")
    grouped = df_passed.groupby("gate_up_cores")["avg_time_us"].agg(["count", "mean", "min"])
    logger.info(grouped.to_string())

    logger.info("\nImpact of down_cores:")
    grouped = df_passed.groupby("down_cores")["avg_time_us"].agg(["count", "mean", "min"])
    logger.info(grouped.to_string())

    # Analyze subblock impact
    logger.info("\nImpact of output subblock size:")
    df_passed["subblock"] = df_passed.apply(lambda row: f"({row['out_subblock_h']},{row['out_subblock_w']})", axis=1)
    grouped = df_passed.groupby("subblock")["avg_time_us"].agg(["count", "mean", "min"])
    logger.info(grouped.to_string())

    # Analyze per_core_M impact
    logger.info("\nImpact of per_core_M:")
    grouped = df_passed.groupby("per_core_M")["avg_time_us"].agg(["count", "mean", "min"])
    logger.info(grouped.to_string())


def print_optimal_config(df: pd.DataFrame):
    """Print the optimal configuration with details."""
    df_passed = df[df["passed"] == True]
    if len(df_passed) == 0:
        return

    best = df_passed.loc[df_passed["avg_time_us"].idxmin()]

    logger.info(f"\n{'=' * 80}")
    logger.info("OPTIMAL CONFIGURATION")
    logger.info(f"{'=' * 80}")
    logger.info(f"\nConfiguration Parameters:")
    logger.info(f"  cores:           {best['cores']}")
    logger.info(f"  in0_block_w:     {best['in0_block_w']}")
    logger.info(f"  out_subblock_h:  {best['out_subblock_h']}")
    logger.info(f"  out_subblock_w:  {best['out_subblock_w']}")
    logger.info(f"  per_core_M:      {best['per_core_M']}")

    logger.info(f"\nProblem Dimensions:")
    logger.info(f"  batch_size:      {best['batch_size']}")
    logger.info(f"  seq_len:         {best['seq_len']}")
    logger.info(f"  hidden_size:     {best['hidden_size']}")
    logger.info(f"  intermediate:    {best['intermediate_size']}")
    logger.info(f"  num_experts:     {best['num_experts_per_device']}")

    logger.info(f"\nPerformance:")
    logger.info(f"  Average time:    {best['avg_time_us']:.2f} us")
    logger.info(f"  Min time:        {best['min_time_us']:.2f} us")
    logger.info(f"  Max time:        {best['max_time_us']:.2f} us")
    logger.info(f"  Std deviation:   {best['std_time_us']:.2f} us")
    logger.info(f"  PCC:             {best['pcc']:.6f}")

    logger.info(f"\nTo apply this configuration, update ThroughputProgramConfig in:")
    logger.info(f"  models/demos/gpt_oss/tt/experts_throughput/config.py")
    logger.info(f"\nWith these values:")
    logger.info(f"  gate_up_cores = {best['cores']}")
    logger.info(f"  down_cores = {best['cores']}")
    logger.info(f"  in0_block_w = {best['in0_block_w']}")
    logger.info(f"  out_subblock_h = {best['out_subblock_h']}")
    logger.info(f"  out_subblock_w = {best['out_subblock_w']}")
    logger.info(f"  per_core_M = {best['per_core_M']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze parameter sweep results for experts matmul")
    parser.add_argument(
        "result_file",
        nargs="?",
        help="Path to result JSON file (default: most recent in param_sweep_results/)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top configurations to display (default: 10)",
    )
    parser.add_argument(
        "--csv",
        help="Export results to CSV file",
    )

    args = parser.parse_args()

    # Load results
    result_file = args.result_file if args.result_file else find_latest_result()
    logger.info(f"Loading results from: {result_file}")

    df = load_results(result_file)

    # Print analysis
    print_summary(df)
    print_best_configs(df, top_n=args.top)
    analyze_parameter_impact(df)
    print_optimal_config(df)

    # Export to CSV if requested
    if args.csv:
        df.to_csv(args.csv, index=False)
        logger.info(f"\nResults exported to: {args.csv}")

    logger.info(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
