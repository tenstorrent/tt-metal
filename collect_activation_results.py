#!/usr/bin/env python3

import os
import csv
import re
import glob
import sys
from pathlib import Path
from datetime import datetime

# Default pattern - can be overridden via command line
DEFAULT_PATTERN = "16_320_320_YOLOV4_HEIGHT_SHARDED_ACT"


def extract_func_name_from_directory(dir_name, pattern_suffix):
    """Extract function name from directory like '16_320_320_YOLOV4_HEIGHT_SHARDED_ACT'"""
    if dir_name.endswith(f"_{pattern_suffix}"):
        return dir_name[: -len(f"_{pattern_suffix}")]
    return dir_name


def get_latest_csv_file(func_dir):
    """Get the most recent CSV file from a function directory"""
    # Find all timestamp subdirectories
    timestamp_dirs = glob.glob(os.path.join(func_dir, "20*"))
    if not timestamp_dirs:
        return None

    # Sort by timestamp (most recent first)
    timestamp_dirs.sort(reverse=True)
    latest_timestamp_dir = timestamp_dirs[0]

    # Find CSV file in the latest timestamp directory
    csv_files = glob.glob(os.path.join(latest_timestamp_dir, "ops_perf_results_*.csv"))
    if csv_files:
        return csv_files[0]

    return None


def extract_dkd_and_attributes(csv_file):
    """Extract DKD and operation attributes from CSV file"""
    try:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Find column indices
            dkd_idx = header.index("DEVICE KERNEL DURATION [ns]")
            attributes_idx = header.index("ATTRIBUTES")
            math_fidelity_idx = header.index("MATH FIDELITY")

            # Read data row
            row = next(reader)

            dkd = row[dkd_idx]
            attributes = row[attributes_idx]
            math_fidelity = row[math_fidelity_idx]

            # Extract operation type from attributes
            op_type_match = re.search(r"op_type=UnaryOpType::(\w+)", attributes)
            actual_op_type = op_type_match.group(1).lower() if op_type_match else "unknown"

            # Check for fast_and_approximate_mode
            fast_approx = (
                "Y"
                if "fast_and_approximate_mode" in attributes
                and "'fast_and_approximate_mode': 'true'" in attributes.lower()
                else "N"
            )

            return {
                "dkd": int(dkd),
                "actual_op_type": actual_op_type,
                "math_fidelity": math_fidelity,
                "fast_approx": fast_approx,
                "csv_file": os.path.basename(csv_file),
                "timestamp": os.path.basename(os.path.dirname(csv_file)),
            }

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None


def analyze_profiling_results(pattern_suffix):
    """Analyze all profiling results for given pattern"""

    base_dir = "/home/ubuntu/logical_not/tt-metal/generated/profiler/reports"

    # Find all directories with the specified pattern
    pattern = os.path.join(base_dir, f"*{pattern_suffix}")
    func_directories = glob.glob(pattern)

    print(f"Found {len(func_directories)} activation function directories with {pattern_suffix} pattern")
    print()

    results = []

    for func_dir in sorted(func_directories):
        dir_name = os.path.basename(func_dir)
        func_name = extract_func_name_from_directory(dir_name, pattern_suffix)

        print(f"Processing: {dir_name}")

        # Get the latest CSV file
        csv_file = get_latest_csv_file(func_dir)

        if csv_file:
            # Extract DKD and other data
            data = extract_dkd_and_attributes(csv_file)

            if data:
                result = {
                    "func_name": func_name,
                    "dir_name": dir_name,
                    "dkd_ns": data["dkd"],
                    "actual_op_type": data["actual_op_type"],
                    "math_fidelity": data["math_fidelity"],
                    "fast_approx": data["fast_approx"],
                    "timestamp": data["timestamp"],
                    "csv_file": data["csv_file"],
                }
                results.append(result)

                print(
                    f"  ✅ DKD: {data['dkd']} ns | Op: {data['actual_op_type']} | Approx: {data['fast_approx']} | Fidelity: {data['math_fidelity']}"
                )
            else:
                print(f"  ❌ Failed to extract data from CSV")
        else:
            print(f"  ❌ No CSV file found")

        print()

    return results


def create_consolidated_report(results, pattern_suffix):
    """Create consolidated CSV report"""

    if not results:
        print("No results to report!")
        return

    # Sort results by DKD (fastest first)
    results.sort(key=lambda x: x["dkd_ns"])

    # Create CSV report
    output_file = f"/home/ubuntu/logical_not/tt-metal/activation_dkd_results_{pattern_suffix}.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            ["Rank", "Func_Name", "DKD_ns", "Actual_Op_Type", "Math_Fidelity", "Timestamp", "Directory_Name"]
        )

        # Write data
        for i, result in enumerate(results, 1):
            writer.writerow(
                [
                    i,
                    result["func_name"],
                    result["dkd_ns"],
                    result["actual_op_type"],
                    result["math_fidelity"],
                    result["timestamp"],
                    result["dir_name"],
                ]
            )

    print(f"✅ Consolidated report saved to: {output_file}")
    return output_file


def print_performance_summary(results):
    """Print performance analysis summary"""

    if not results:
        return

    print("=" * 80)
    print("ACTIVATION FUNCTION PERFORMANCE SUMMARY (32x32 L1 ACT)")
    print("=" * 80)

    print(f"Total Functions Analyzed: {len(results)}")
    print()

    # Performance ranking
    print("PERFORMANCE RANKING (by DKD):")
    print("Rank | Func_Name                | DKD (ns) | Op_Type          | Math_Fidelity")
    print("-" * 75)

    for i, result in enumerate(results[:15], 1):  # Show top 15
        print(
            f"{i:4d} | {result['func_name']:<24} | {result['dkd_ns']:8d} | {result['actual_op_type']:<16} | {result['math_fidelity']}"
        )

    if len(results) > 15:
        print(f"... and {len(results) - 15} more entries")

    print()

    # Find Mish position
    mish_results = [r for r in results if "mish" in r["func_name"].lower()]
    if mish_results:
        mish_result = mish_results[0]
        mish_rank = results.index(mish_result) + 1
        print(f"🎯 MISH PERFORMANCE:")
        print(f"   Rank: {mish_rank}/{len(results)}")
        print(f"   DKD:  {mish_result['dkd_ns']} ns")
        print(f"   Type: {mish_result['actual_op_type']}")
        print()

    # Performance categories
    fast_ops = [r for r in results if r["dkd_ns"] < 2000]
    medium_ops = [r for r in results if 2000 <= r["dkd_ns"] < 4000]
    slow_ops = [r for r in results if r["dkd_ns"] >= 4000]

    print("PERFORMANCE CATEGORIES:")
    print(f"Fast (< 2000ns):     {len(fast_ops):2d} functions")
    print(f"Medium (2000-4000ns):{len(medium_ops):2d} functions")
    print(f"Slow (≥ 4000ns):     {len(slow_ops):2d} functions")
    print()

    # Approximate vs Accurate comparison
    approx_ops = [r for r in results if r["fast_approx"] == "Y"]
    accurate_ops = [r for r in results if r["fast_approx"] == "N"]

    print("APPROXIMATION MODE ANALYSIS:")
    print(f"Accurate mode:   {len(accurate_ops):2d} functions")
    print(f"Approximate mode:{len(approx_ops):2d} functions")

    if approx_ops:
        print("\nApproximate mode functions:")
        for r in approx_ops:
            print(f"  {r['func_name']}: {r['dkd_ns']} ns")


def main():
    # Get pattern from command line argument or use default
    pattern_suffix = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATTERN

    print(f"Collecting DKD results for all {pattern_suffix} activation functions...")
    print()

    # Analyze all results
    results = analyze_profiling_results(pattern_suffix)

    if results:
        # Create consolidated report
        output_file = create_consolidated_report(results, pattern_suffix)

        # Print summary
        print_performance_summary(results)

        print()
        print(f"📊 Detailed results saved to: {output_file}")

    else:
        print("❌ No results found!")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python collect_activation_results.py [PATTERN_SUFFIX]")
        print(f"Example: python collect_activation_results.py 32_32_L1_ACT")
        print(f"Example: python collect_activation_results.py 64_128_L1_ACT")
        print(f"Default: {DEFAULT_PATTERN}")
        sys.exit(1)
    main()
