#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Utilization Report Generator

Usage: gen_util_report.py -o <output_dir> -c <command>

Steps:
  1. Run tracy twice: once with NOC traces, once with perf counters
  2. Find and read the resulting CSV files
  3. Clean data (remove signposts, invalid rows) and merge on GLOBAL CALL COUNT
  4. Extract performance metrics and save final report

Examples:
  python gen_util_report.py -o ./results -c "python my_benchmark.py"
  python gen_util_report.py -o ./output --skip-profiling  # reprocess existing data
"""
import argparse
import os
import sys
import subprocess
import glob
import pandas as pd
import re


# Run tracy with specified options
def run_profile_command(command, output_dir, subdir, profile_options):
    full_output_path = os.path.join(output_dir, subdir)
    profile_cmd = (
        [sys.executable, "-m", "tracy", "-v", "-r", "-p", "-o", full_output_path] + profile_options + ["-m", command]
    )

    subprocess.run(profile_cmd, check=True)
    return full_output_path


# Find latest ops_perf_results_*.csv file in timestamped subdirectories
def find_csv_files(directory):
    from datetime import datetime

    csv_files = glob.glob(os.path.join(directory, "**/ops_perf_results_*.csv"), recursive=True)

    # Parse timestamps from directory names and sort by latest
    def get_timestamp(f):
        match = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", os.path.basename(os.path.dirname(f)))
        return datetime.strptime(match.group(1), "%Y_%m_%d_%H_%M_%S") if match else datetime.min

    return sorted(csv_files, key=get_timestamp, reverse=True)[:1] or csv_files


# Remove rows with op type "signpost" or those without a "DEVICE KERNEL DURATION [ns]" value
def process_cleanup_data(df):
    df = df[df["OP TYPE"] != "signpost"]
    df = df[df["DEVICE KERNEL DURATION [ns]"].notna()]

    if len(df) == 0:
        return None

    return df


# Merge perf dataframe with NPE dataframe on GLOBAL CALL COUNT and METAL TRACE REPLAY SESSION ID.
# Takes NPE util columns from npe_df and everything else from perf_df.
def merge_dataframes(perf_df, npe_df):
    merge_keys = ["GLOBAL CALL COUNT", "METAL TRACE REPLAY SESSION ID"]
    npe_cols = ["NOC UTIL (%)", "DRAM BW UTIL (%)", "NPE CONG IMPACT (%)"]

    # Get NPE columns that exist
    npe_cols_to_use = [c for c in npe_cols if c in npe_df.columns]
    npe_subset = npe_df[merge_keys + npe_cols_to_use].copy()

    # Drop NPE columns from perf_df if they exist (we'll take them from npe_df)
    perf_df = perf_df.drop(columns=[c for c in npe_cols_to_use if c in perf_df.columns], errors="ignore")

    # Merge
    merged_df = perf_df.merge(npe_subset, on=merge_keys, how="inner")

    return merged_df


# Filter dataframe to keep only rows with the highest METAL TRACE REPLAY SESSION ID."""
def filter_highest_session_id(df):
    col = "METAL TRACE REPLAY SESSION ID"
    df[col] = pd.to_numeric(df[col], errors="coerce")
    max_id = df[col].max()
    return df[df[col] == max_id]


def extract_logical_value(value):
    """Extract logical value from format like '640[640]' -> 640"""
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, str) and "[" in value and "]" in value:
        return int(value.split("[")[1].split("]")[0])
    return int(float(value))


def extract_performance_metrics(df):
    """Extract and reorder performance metrics columns from dataframe."""
    target_columns = [
        "OP CODE",
        "GLOBAL CALL COUNT",
        "CORE COUNT",
        "NOC UTIL (%)",
        "DRAM BW UTIL (%)",
        "NPE CONG IMPACT (%)",
        "PM COMPUTE [ns]",
        "PM FPU UTIL (%)",
        "DEVICE KERNEL DURATION [ns]",
        "Packet Size Min",
        "Packet Size Q1",
        "Packet Size Median",
        "Packet Size Q3",
        "Packet Size Max",
        "SFPU Util Min (%)",
        "SFPU Util Median (%)",
        "SFPU Util Max (%)",
        "Avg SFPU util on full grid (%)",
        "FPU Util Min (%)",
        "FPU Util Median (%)",
        "FPU Util Max (%)",
        "Avg FPU util on full grid (%)",
        "MATH Util Min (%)",
        "MATH Util Median (%)",
        "MATH Util Max (%)",
        "Avg Math util on full grid (%)",
    ]

    # Extract existing columns
    existing_cols = [c for c in target_columns if c in df.columns]
    extracted_df = df[existing_cols].copy()

    # Add % of Total Cycles
    if "DEVICE KERNEL DURATION [ns]" in existing_cols:
        total = extracted_df["DEVICE KERNEL DURATION [ns]"].sum()
        if total > 0:
            extracted_df["% of Total Cycles"] = (extracted_df["DEVICE KERNEL DURATION [ns]"] / total) * 100

    # Add logical size and mem config columns for each input/output
    for prefix in ["INPUT_0", "INPUT_1", "INPUT_2", "OUTPUT_0"]:
        # Logical size: [W, Z, Y, X]
        pad_cols = [f"{prefix}_{d}_PAD[LOGICAL]" for d in ["W", "Z", "Y", "X"]]
        if all(c in df.columns for c in pad_cols):

            def make_size(row, cols=pad_cols):
                try:
                    vals = [extract_logical_value(row[c]) for c in cols]
                    return (
                        f"[{vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}]" if all(v is not None for v in vals) else "N/A"
                    )
                except:
                    return "N/A"

            extracted_df[f"{prefix}_LOGICAL_SIZE"] = df.apply(make_size, axis=1)

        # Mem config: LAYOUT-DATATYPE-MEMORY
        mem_cols = [f"{prefix}_{x}" for x in ["LAYOUT", "DATATYPE", "MEMORY"]]
        if all(c in df.columns for c in mem_cols):

            def make_mem(row, cols=mem_cols):
                try:
                    vals = [str(row[c]) for c in cols if pd.notna(row[c]) and row[c] != ""]
                    return "-".join(vals) if len(vals) == 3 else "N/A"
                except:
                    return "N/A"

            extracted_df[f"{prefix}_MEM_CONFIG"] = df.apply(make_mem, axis=1)

    # Round percentage columns
    pct_cols = [c for c in extracted_df.columns if "(%)" in c or c == "% of Total Cycles"]
    for col in pct_cols:
        if extracted_df[col].dtype in ["float64", "float32"]:
            extracted_df[col] = extracted_df[col].round(3)

    return extracted_df


def main():
    parser = argparse.ArgumentParser(description="Generate utilization reports")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for reports")
    parser.add_argument("-c", "--command", required=True, help="Command to profile")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip profiling, use existing CSV files")
    parser.add_argument("--single-model-iteration", action="store_true", help="Filter to highest session ID only")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.abspath(args.output_dir)

    print("Step 1: Running profiling...")
    if not args.skip_profiling:
        npe_dir = run_profile_command(
            args.command, output_dir, "perf_report_with_npe_metrics", ["--collect-noc-traces"]
        )
        perf_dir = run_profile_command(
            args.command, output_dir, "perf_report", ["--profiler-capture-perf-counters=fpu"]
        )
    else:
        print("  Skipping (using existing files)")
        npe_dir = os.path.join(output_dir, "perf_report_with_npe_metrics")
        perf_dir = os.path.join(output_dir, "perf_report")

    print("Step 2: Reading CSV files...")
    npe_df = pd.read_csv(find_csv_files(npe_dir)[0])
    perf_df = pd.read_csv(find_csv_files(perf_dir)[0])

    print("Step 3: Cleaning and merging data...")
    merged_df = merge_dataframes(process_cleanup_data(perf_df), process_cleanup_data(npe_df))

    if args.single_model_iteration:
        merged_df = filter_highest_session_id(merged_df)

    print("Step 4: Extracting metrics and saving...")
    final_df = extract_performance_metrics(merged_df)
    final_df.to_csv(os.path.join(output_dir, "model_util_report.csv"), index=False)
    print(f"Done! Report saved to {output_dir}/model_util_report.csv")


if __name__ == "__main__":
    main()
