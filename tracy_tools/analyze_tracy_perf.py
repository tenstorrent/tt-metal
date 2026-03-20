#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Analyze Tracy profiler CSV outputs and generate performance summaries.

This script processes ops_perf_results CSV files from Tracy profiling to calculate
and display summary statistics including average, min, and max kernel durations.

Usage:
    # Analyze most recent Tracy CSV
    python analyze_tracy_perf.py

    # Analyze specific CSV file
    python analyze_tracy_perf.py --csv path/to/ops_perf_results.csv

    # Analyze most recent CSV in a specific directory
    python analyze_tracy_perf.py --dir generated/profiler/reports/

    # Save output to custom CSV file
    python analyze_tracy_perf.py --output my_summary.csv

    # Run Tracy profiling and analyze results
    python analyze_tracy_perf.py --run "python test_script.py"
    python analyze_tracy_perf.py --run "pytest test_file.py::test_name"

    # Run with --no-runtime-analysis for full per-core data (e.g. DEVICE COMPUTE CB WAIT FRONT)
    python analyze_tracy_perf.py --run "python test_script.py" --no-runtime-analysis
"""

import os
import sys
import argparse
import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple


IMPL_ORDER = [
    "RMSNorm[Tile]",
    "RMSNormComposite",
    "RMSNormFused+CompositeTilizing",
    "RMSNormFullFusion",
]


def get_all_tracy_csvs(search_dir: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Get all ops_perf_results CSV files with their modification times.

    Args:
        search_dir: Directory to search in. If None, uses $TT_METAL_HOME/generated/profiler/reports/

    Returns:
        List of tuples (csv_path, modification_time)
    """
    if search_dir is None:
        tt_metal_home = os.environ.get("TT_METAL_HOME")
        if not tt_metal_home:
            tt_metal_home = "/localdev/nmaurice/tt-metal"
        search_dir = os.path.join(tt_metal_home, "generated", "profiler", "reports")

    search_path = Path(search_dir)
    if not search_path.exists():
        return []

    # Find all ops_perf_results CSV files recursively
    csv_files = list(search_path.glob("**/ops_perf_results*.csv"))

    # Return with modification times
    return [(str(f), f.stat().st_mtime) for f in csv_files]


def find_latest_tracy_csv(search_dir: Optional[str] = None) -> Optional[str]:
    """
    Find the most recent ops_perf_results CSV file.

    Args:
        search_dir: Directory to search in. If None, uses $TT_METAL_HOME/generated/profiler/reports/

    Returns:
        Path to the most recent CSV file, or None if not found
    """
    csv_list = get_all_tracy_csvs(search_dir)

    if not csv_list:
        if search_dir is None:
            tt_metal_home = os.environ.get("TT_METAL_HOME")
            if not tt_metal_home:
                tt_metal_home = "/localdev/nmaurice/tt-metal"
            search_dir = os.path.join(tt_metal_home, "generated", "profiler", "reports")
        print(f"Error: No ops_perf_results*.csv files found in {search_dir}", file=sys.stderr)
        return None

    # Sort by modification time, most recent first
    csv_list.sort(key=lambda x: x[1], reverse=True)

    return csv_list[0][0]


def run_tracy_command(
    command: str,
    tracy_output_dir: Optional[str] = None,
    no_runtime_analysis: bool = False,
) -> Optional[str]:
    """
    Run a command with Tracy profiling and return the path to the generated CSV.

    Args:
        command: Command to run (e.g., "python script.py" or "pytest test.py::test_name")
        tracy_output_dir: Directory for Tracy output. If None, uses $TT_METAL_HOME/generated/profiler/reports
        no_runtime_analysis: Pass --no-runtime-analysis to tracy, disabling C++ post-processing.
            This produces fuller per-core data (e.g. DEVICE COMPUTE CB WAIT FRONT) useful for
            determining compute vs. memory boundedness.

    Returns:
        Path to the generated ops_perf_results CSV file, or None if not found
    """
    if tracy_output_dir is None:
        tt_metal_home = os.environ.get("TT_METAL_HOME")
        if not tt_metal_home:
            tt_metal_home = "/localdev/nmaurice/tt-metal"
        tracy_output_dir = os.path.join(tt_metal_home, "generated", "profiler", "reports")

    # Get list of existing CSV files before running
    existing_csvs = set(path for path, _ in get_all_tracy_csvs(tracy_output_dir))

    # Build Tracy command
    tracy_cmd = [
        "python",
        "-m",
        "tracy",
        "-r",  # Generate ops report
        "-p",  # Partial profile
        "-v",  # Verbose
        "-m",  # Module mode
    ]

    if no_runtime_analysis:
        tracy_cmd.append("--no-runtime-analysis")

    # Parse the command
    # Handle pytest-style commands
    if command.startswith("pytest ") or "::" in command:
        # pytest command
        if command.startswith("pytest "):
            test_path = command[7:].strip()  # Remove "pytest "
        else:
            # Assume it's "python file.py::test_name" format
            if command.startswith("python "):
                test_path = command[7:].strip()  # Remove "python "
            else:
                test_path = command.strip()

        tracy_cmd.extend(["pytest", test_path])
    else:
        # Regular python command
        if command.startswith("python "):
            cmd_parts = command[7:].strip().split()
        else:
            cmd_parts = command.strip().split()

        tracy_cmd.extend(cmd_parts)

    print(f"Running Tracy command: {' '.join(tracy_cmd)}")
    print("-" * 80)

    print(f"tracy command: {' '.join(tracy_cmd)}")

    # Run Tracy
    try:
        result = subprocess.run(tracy_cmd, check=True, capture_output=False, text=True)  # Show output in real-time
    except subprocess.CalledProcessError as e:
        print(f"\nError: Tracy command failed with exit code {e.returncode}", file=sys.stderr)
        return None

    print("-" * 80)
    print("Tracy profiling completed")

    # Wait a moment for CSV to be written
    time.sleep(1)

    # Find the new CSV file
    current_csvs = set(path for path, _ in get_all_tracy_csvs(tracy_output_dir))
    new_csvs = current_csvs - existing_csvs

    if not new_csvs:
        print("Warning: No new CSV file detected. Using most recent CSV.", file=sys.stderr)
        return find_latest_tracy_csv(tracy_output_dir)

    if len(new_csvs) > 1:
        print(f"Warning: Multiple new CSV files found, using most recent one", file=sys.stderr)
        # Get the most recent one
        new_csv_list = [(path, Path(path).stat().st_mtime) for path in new_csvs]
        new_csv_list.sort(key=lambda x: x[1], reverse=True)
        return new_csv_list[0][0]

    return list(new_csvs)[0]


def parse_shape_column(shape_str: str) -> int:
    """
    Parse shape column value like '32768[32760]' to extract the logical dimension.

    Args:
        shape_str: Shape string from CSV (e.g., '32768[32760]')

    Returns:
        Logical dimension value (e.g., 32760)
    """
    if pd.isna(shape_str) or shape_str == "":
        return 0

    # Extract the value inside brackets
    if "[" in shape_str and "]" in shape_str:
        start = shape_str.index("[")
        end = shape_str.index("]")
        return int(shape_str[start + 1 : end])

    # If no brackets, try to parse as integer
    try:
        return int(shape_str)
    except:
        return 0


def format_shape(w: int, z: int, y: int, x: int) -> str:
    """
    Format shape dimensions into a readable string.

    Args:
        w, z, y, x: Shape dimensions

    Returns:
        Formatted shape string (e.g., '[1, 1, 32760, 5120]')
    """
    # Only include non-1 dimensions for readability
    dims = [w, z, y, x]

    # If all dimensions are present, show full shape
    if all(d > 0 for d in dims):
        # Simplify by removing leading 1s
        while len(dims) > 2 and dims[0] == 1:
            dims = dims[1:]
        return f"[{', '.join(map(str, dims))}]"

    return "[]"


def parse_shape_from_config_name(config_name: str):
    """
    If config_name is in benchmark format (e.g. "bf16/1x4x480x832x96"), parse it
    and return a shape string for display. Otherwise return None.
    Used so the summary shows the correct logical shape when the device profiler
    reports a stale/cached shape (e.g. BinaryNg always showing the first program's shape).
    """
    if not config_name or "/" not in config_name:
        return None
    _, dims_str = config_name.split("/", 1)
    try:
        dims = [int(d) for d in dims_str.split("x")]
    except (ValueError, AttributeError):
        return None
    if not dims:
        return None
    # Drop leading 1s for readability, like format_shape
    while len(dims) > 2 and dims[0] == 1:
        dims = dims[1:]
    return "[" + ", ".join(map(str, dims)) + "]"


def extract_op_name(op_code: str) -> str:
    """
    Extract a clean operation name from the OP CODE column.

    Args:
        op_code: Value from OP CODE column

    Returns:
        Cleaned operation name
    """
    # Remove common suffixes like "DeviceOperation"
    name = op_code.replace("DeviceOperation", "").replace("Operation", "")
    return name.strip()


def analyze_tracy_csv_aggregated(csv_path: str) -> pd.DataFrame:
    """
    Analyze a Tracy profiler CSV file with aggregation by implementation.

    Only includes operations between signpost markers (excludes warmup).
    For composite operations, sums constituent kernels.
    Returns one row per implementation with averaged metrics.

    Args:
        csv_path: Path to the ops_perf_results CSV file

    Returns:
        DataFrame with aggregated performance summary (one row per implementation)
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        return pd.DataFrame()

    # Check if required columns exist
    required_cols = ["OP CODE"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}", file=sys.stderr)
        return pd.DataFrame()

    # Check if DEVICE KERNEL DURATION column exists
    has_device_duration = "DEVICE KERNEL DURATION [ns]" in df.columns

    # Detect whether CB wait data is present (populated by --no-runtime-analysis)
    _cb_col = "DEVICE COMPUTE CB WAIT FRONT [ns]"
    has_cb_wait = _cb_col in df.columns and df[_cb_col].apply(lambda v: not pd.isna(v) and v != "" and v != 0).any()

    # Track state
    current_implementation = None
    in_measurement = False
    implementations = {}  # {impl_name: {ops: [list of all ops in measurement]}}

    for _, row in df.iterrows():
        op_code = row["OP CODE"]

        # Skip empty rows
        if pd.isna(op_code):
            continue

        # Handle signposts
        if row.get("OP TYPE") == "signpost":
            if "-start" in op_code:
                # Start of measurement section
                current_implementation = op_code.replace("-start", "")
                in_measurement = True
                if current_implementation not in implementations:
                    implementations[current_implementation] = {"ops": []}
            elif "-end" in op_code:
                # End of measurement section
                in_measurement = False
                current_implementation = None
            continue

        # Only process operations during measurement phase
        if not in_measurement or current_implementation is None:
            continue

        # Extract duration
        duration_ns = None
        if has_device_duration:
            duration_ns = row.get("DEVICE KERNEL DURATION [ns]")

        if pd.isna(duration_ns) or duration_ns is None:
            continue

        # Extract CB wait time when available
        cb_wait_ns = None
        if has_cb_wait:
            raw = row.get(_cb_col)
            if not pd.isna(raw) and raw != "":
                cb_wait_ns = float(raw)

        # Extract shape information
        output_w = parse_shape_column(row.get("OUTPUT_0_W_PAD[LOGICAL]", ""))
        output_z = parse_shape_column(row.get("OUTPUT_0_Z_PAD[LOGICAL]", ""))
        output_y = parse_shape_column(row.get("OUTPUT_0_Y_PAD[LOGICAL]", ""))
        output_x = parse_shape_column(row.get("OUTPUT_0_X_PAD[LOGICAL]", ""))
        shape_str = format_shape(output_w, output_z, output_y, output_x)

        # Extract operation name
        op_name = extract_op_name(op_code)

        # Collect all operations
        implementations[current_implementation]["ops"].append(
            {
                "op_name": op_name,
                "shape": shape_str,
                "duration_ns": duration_ns,
                "cb_wait_ns": cb_wait_ns,
            }
        )

    # Process collected data
    results = []

    for impl_name, impl_data in implementations.items():
        all_ops = impl_data["ops"]

        if not all_ops:
            continue

        # Identify unique operation types (by name)
        unique_op_names = []
        seen_names = set()
        for op in all_ops:
            if op["op_name"] not in seen_names:
                unique_op_names.append(op["op_name"])
                seen_names.add(op["op_name"])

        num_unique_ops = len(unique_op_names)
        total_ops = len(all_ops)

        if num_unique_ops == 0:
            continue

        # Determine number of iterations
        num_iterations = total_ops // num_unique_ops

        if num_iterations == 0:
            continue

        if num_unique_ops == 1:
            # Single operation type - each operation is one iteration
            durations = [op["duration_ns"] for op in all_ops]

            avg_duration_ms = sum(durations) / len(durations) / 1_000_000
            min_duration_ms = min(durations) / 1_000_000
            max_duration_ms = max(durations) / 1_000_000
            std_duration_ms = np.std(durations) / 1_000_000 if len(durations) > 1 else 0.0

            cb_waits = [op["cb_wait_ns"] for op in all_ops if op.get("cb_wait_ns") is not None]
            avg_cb_wait_pct = (
                (sum(cb_waits) / len(cb_waits)) / (sum(durations) / len(durations)) * 100
                if cb_waits and sum(durations) > 0
                else None
            )

            config_name, impl_type = parse_implementation(impl_name)
            shape = parse_shape_from_config_name(config_name) or all_ops[0]["shape"]
            row_dict = {
                "config_name": config_name,
                "impl_type": impl_type,
                "implementation": impl_name,
                "op_name": all_ops[0]["op_name"],
                "shape": shape,
                "avg_duration_ms": f"{avg_duration_ms:.3f}",
                "std_duration_ms": f"{std_duration_ms:.3f}",
                "min_duration_ms": f"{min_duration_ms:.3f}",
                "max_duration_ms": f"{max_duration_ms:.3f}",
            }
            if avg_cb_wait_pct is not None:
                row_dict["cb_wait_pct"] = f"{avg_cb_wait_pct:.1f}"
            results.append(row_dict)
        else:
            # Composite operation - multiple operation types per iteration
            # Group operations into iterations
            iteration_totals = []
            iteration_cb_waits = []

            for i in range(num_iterations):
                iteration_ops = all_ops[i * num_unique_ops : (i + 1) * num_unique_ops]
                total_duration = sum(op["duration_ns"] for op in iteration_ops)
                iteration_totals.append(total_duration)
                cb_vals = [op["cb_wait_ns"] for op in iteration_ops if op.get("cb_wait_ns") is not None]
                if cb_vals:
                    iteration_cb_waits.append(sum(cb_vals))

            if iteration_totals:
                avg_duration_ms = sum(iteration_totals) / len(iteration_totals) / 1_000_000
                min_duration_ms = min(iteration_totals) / 1_000_000
                max_duration_ms = max(iteration_totals) / 1_000_000
                std_duration_ms = np.std(iteration_totals) / 1_000_000 if len(iteration_totals) > 1 else 0.0

                avg_cb_wait_pct = (
                    (sum(iteration_cb_waits) / len(iteration_cb_waits))
                    / (sum(iteration_totals) / len(iteration_totals))
                    * 100
                    if iteration_cb_waits and sum(iteration_totals) > 0
                    else None
                )

                # Create composite name
                op_name = "+".join(unique_op_names)
                config_name, impl_type = parse_implementation(impl_name)
                shape = parse_shape_from_config_name(config_name) or all_ops[0]["shape"]
                row_dict = {
                    "config_name": config_name,
                    "impl_type": impl_type,
                    "implementation": impl_name,
                    "op_name": op_name,
                    "shape": shape,
                    "avg_duration_ms": f"{avg_duration_ms:.3f}",
                    "std_duration_ms": f"{std_duration_ms:.3f}",
                    "min_duration_ms": f"{min_duration_ms:.3f}",
                    "max_duration_ms": f"{max_duration_ms:.3f}",
                }
                if avg_cb_wait_pct is not None:
                    row_dict["cb_wait_pct"] = f"{avg_cb_wait_pct:.1f}"
                results.append(row_dict)

    if not results:
        print("Warning: No measurement data found (no signpost markers?)", file=sys.stderr)
        return pd.DataFrame()

    return pd.DataFrame(results)


def analyze_tracy_csv(csv_path: str, include_signposts: bool = True) -> pd.DataFrame:
    """
    Analyze a Tracy profiler CSV file and extract performance statistics.

    Args:
        csv_path: Path to the ops_perf_results CSV file
        include_signposts: Whether to include signpost markers in the output

    Returns:
        DataFrame with performance summary
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        return pd.DataFrame()

    # Check if required columns exist
    required_cols = ["OP CODE"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}", file=sys.stderr)
        return pd.DataFrame()

    # Detect which format we're dealing with
    has_per_core_cols = all(
        col in df.columns
        for col in [
            "DEVICE KERNEL DURATION PER CORE MIN [ns]",
            "DEVICE KERNEL DURATION PER CORE MAX [ns]",
            "DEVICE KERNEL DURATION PER CORE AVG [ns]",
        ]
    )

    has_individual_core_cols = all(
        col in df.columns
        for col in [
            "DEVICE BRISC KERNEL DURATION [ns]",
            "DEVICE NCRISC KERNEL DURATION [ns]",
            "DEVICE TRISC0 KERNEL DURATION [ns]",
        ]
    )

    # Detect whether CB wait data is present (populated by --no-runtime-analysis)
    _cb_col = "DEVICE COMPUTE CB WAIT FRONT [ns]"
    has_cb_wait = _cb_col in df.columns and df[_cb_col].apply(lambda v: not pd.isna(v) and v != "" and v != 0).any()

    # Process each row
    results = []

    for _, row in df.iterrows():
        op_code = row["OP CODE"]

        # Skip empty rows
        if pd.isna(op_code):
            continue

        # Handle signposts separately
        if row.get("OP TYPE") == "signpost":
            if include_signposts:
                results.append(
                    {
                        "op_name": op_code,
                        "shape": "-",
                        "avg_duration_ms": "-",
                        "min_duration_ms": "-",
                        "max_duration_ms": "-",
                    }
                )
            continue

        # Determine if this row has duration data
        has_duration_data = False
        min_duration_ns = None
        max_duration_ns = None
        avg_duration_ns = None

        # Try new format first (aggregated per-core stats)
        if has_per_core_cols:
            min_val = row.get("DEVICE KERNEL DURATION PER CORE MIN [ns]")
            max_val = row.get("DEVICE KERNEL DURATION PER CORE MAX [ns]")
            avg_val = row.get("DEVICE KERNEL DURATION PER CORE AVG [ns]")

            if not pd.isna(min_val) and min_val > 0:
                has_duration_data = True
                min_duration_ns = min_val
                max_duration_ns = max_val
                avg_duration_ns = avg_val

        # Fall back to old format if new format has no data
        if not has_duration_data and has_individual_core_cols:
            # Old format: has individual core durations, compute stats
            core_durations = []
            for col in [
                "DEVICE BRISC KERNEL DURATION [ns]",
                "DEVICE NCRISC KERNEL DURATION [ns]",
                "DEVICE TRISC0 KERNEL DURATION [ns]",
                "DEVICE TRISC1 KERNEL DURATION [ns]",
                "DEVICE TRISC2 KERNEL DURATION [ns]",
            ]:
                if col in df.columns:
                    val = row.get(col)
                    if not pd.isna(val) and val > 0:
                        core_durations.append(val)

            if core_durations:
                has_duration_data = True
                min_duration_ns = min(core_durations)
                max_duration_ns = max(core_durations)
                avg_duration_ns = sum(core_durations) / len(core_durations)

        # Skip rows without duration data
        if not has_duration_data:
            continue

        # Extract shape information
        output_w = parse_shape_column(row.get("OUTPUT_0_W_PAD[LOGICAL]", ""))
        output_z = parse_shape_column(row.get("OUTPUT_0_Z_PAD[LOGICAL]", ""))
        output_y = parse_shape_column(row.get("OUTPUT_0_Y_PAD[LOGICAL]", ""))
        output_x = parse_shape_column(row.get("OUTPUT_0_X_PAD[LOGICAL]", ""))

        shape_str = format_shape(output_w, output_z, output_y, output_x)

        # Convert durations from ns to ms
        min_duration_ms = min_duration_ns / 1_000_000 if min_duration_ns is not None else 0
        max_duration_ms = max_duration_ns / 1_000_000 if max_duration_ns is not None else 0
        avg_duration_ms = avg_duration_ns / 1_000_000 if avg_duration_ns is not None else 0

        # Extract clean operation name
        op_name = extract_op_name(op_code)

        row_dict = {
            "op_name": op_name,
            "shape": shape_str,
            "avg_duration_ms": f"{avg_duration_ms:.3f}",
            "min_duration_ms": f"{min_duration_ms:.3f}",
            "max_duration_ms": f"{max_duration_ms:.3f}",
        }

        # Compute CB wait % when data is available
        if has_cb_wait:
            raw_cb = row.get(_cb_col)
            if not pd.isna(raw_cb) and raw_cb != "" and avg_duration_ns and avg_duration_ns > 0:
                cb_wait_pct = float(raw_cb) / avg_duration_ns * 100
                row_dict["cb_wait_pct"] = f"{cb_wait_pct:.1f}"
            else:
                row_dict["cb_wait_pct"] = "-"

        results.append(row_dict)

    if not results:
        print("Warning: No operation data found in CSV", file=sys.stderr)
        return pd.DataFrame()

    return pd.DataFrame(results)


def parse_implementation(impl_str: str) -> tuple:
    """
    Parse implementation string to extract config_name and implementation type.

    Args:
        impl_str: Implementation string (e.g., "Fused-wan2_2_14b_480p", "MinimalMatmulBias-wan2_2_14b_480p")

    Returns:
        Tuple of (config_name, impl_type)
    """
    # Known implementation type prefixes
    known_types = ["Fused", "Separate", "MinimalMatmulBias", "MinimalMatmul", "Addcmul"]

    # Try to match against known types
    for impl_type in known_types:
        if impl_str.startswith(impl_type + "-"):
            config_name = impl_str[len(impl_type) + 1 :]  # Skip the dash
            return config_name, impl_type

    # Fallback: try to split on first dash
    parts = impl_str.split("-", 1)
    if len(parts) == 2:
        impl_type = parts[0]
        config_name = parts[1]
    else:
        # No dash found
        impl_type = impl_str
        config_name = "unknown"

    return config_name, impl_type


def print_summary_table(df: pd.DataFrame):
    """
    Print a formatted summary table to console.

    Args:
        df: DataFrame with performance summary
    """
    if df.empty:
        print("No data to display")
        return

    # Check if this is aggregated format (has implementation column)
    is_aggregated = "implementation" in df.columns

    # Whether CB wait % column is present
    has_cb_col = "cb_wait_pct" in df.columns

    if is_aggregated:
        # Parse implementation strings to extract config_name and impl_type
        df_copy = df.copy()
        df_copy["config_name"] = df_copy["implementation"].apply(lambda x: parse_implementation(x)[0])
        df_copy["impl_type"] = df_copy["implementation"].apply(lambda x: parse_implementation(x)[1])

        # Sort by config_name, then impl_type using explicit IMPL_ORDER (unknown types sort last alphabetically)
        known = [t for t in IMPL_ORDER if t in df_copy["impl_type"].values]
        unknown = sorted(t for t in df_copy["impl_type"].unique() if t not in IMPL_ORDER)
        ordered_types = known + unknown
        df_copy["impl_type"] = pd.Categorical(df_copy["impl_type"], categories=ordered_types, ordered=True)
        df_copy = df_copy.sort_values(["config_name", "impl_type"])

        # Group by config_name
        grouped = df_copy.groupby("config_name")

        # Calculate column widths
        col_widths = {
            "config_impl": max(
                len("Config: Implementation"),
                max(len(f"{row['config_name']}: {row['impl_type']}") for _, row in df_copy.iterrows()),
            ),
            "op_name": max(len("Operation"), max(len(str(x)) for x in df_copy["op_name"])),
            "shape": max(len("Shape"), max(len(str(x)) for x in df_copy["shape"])),
            "avg_duration_ms": max(len("Avg (ms)"), max(len(str(x)) for x in df_copy["avg_duration_ms"])),
            "std_duration_ms": max(len("Std (ms)"), max(len(str(x)) for x in df_copy["std_duration_ms"])),
            "min_duration_ms": max(len("Min (ms)"), max(len(str(x)) for x in df_copy["min_duration_ms"])),
            "max_duration_ms": max(len("Max (ms)"), max(len(str(x)) for x in df_copy["max_duration_ms"])),
        }
        if has_cb_col:
            col_widths["cb_wait_pct"] = max(len("CB Wait (%)"), max(len(str(x)) for x in df_copy["cb_wait_pct"]))

        # Print header
        header = (
            f"{'Config: Implementation':<{col_widths['config_impl']}} | "
            f"{'Operation':<{col_widths['op_name']}} | "
            f"{'Shape':<{col_widths['shape']}} | "
            f"{'Avg (ms)':>{col_widths['avg_duration_ms']}} | "
            f"{'Std (ms)':>{col_widths['std_duration_ms']}} | "
            f"{'Min (ms)':>{col_widths['min_duration_ms']}} | "
            f"{'Max (ms)':>{col_widths['max_duration_ms']}}"
        )
        if has_cb_col:
            header += f" | {'CB Wait (%)':>{col_widths['cb_wait_pct']}}"
        separator = "=" * len(header)

        print()
        print(separator)
        print(header)
        print(separator)

        # Print rows grouped by config
        for config_name, group in grouped:
            for _, row in group.iterrows():
                config_impl = f"{row['config_name']}: {row['impl_type']}"
                line = (
                    f"{config_impl:<{col_widths['config_impl']}} | "
                    f"{row['op_name']:<{col_widths['op_name']}} | "
                    f"{row['shape']:<{col_widths['shape']}} | "
                    f"{row['avg_duration_ms']:>{col_widths['avg_duration_ms']}} | "
                    f"{row['std_duration_ms']:>{col_widths['std_duration_ms']}} | "
                    f"{row['min_duration_ms']:>{col_widths['min_duration_ms']}} | "
                    f"{row['max_duration_ms']:>{col_widths['max_duration_ms']}}"
                )
                if has_cb_col:
                    line += f" | {str(row.get('cb_wait_pct', '-')):>{col_widths['cb_wait_pct']}}"
                print(line)
            # Add a blank line between config groups
            print()

        print(separator)
        print()
    else:
        # Original format without implementation column
        col_widths = {
            "op_name": max(len("Operation"), max(len(str(x)) for x in df["op_name"])),
            "shape": max(len("Shape"), max(len(str(x)) for x in df["shape"])),
            "avg_duration_ms": max(len("Avg (ms)"), max(len(str(x)) for x in df["avg_duration_ms"])),
            "min_duration_ms": max(len("Min (ms)"), max(len(str(x)) for x in df["min_duration_ms"])),
            "max_duration_ms": max(len("Max (ms)"), max(len(str(x)) for x in df["max_duration_ms"])),
        }
        if has_cb_col:
            col_widths["cb_wait_pct"] = max(len("CB Wait (%)"), max(len(str(x)) for x in df["cb_wait_pct"]))

        # Print header
        header = (
            f"{'Operation':<{col_widths['op_name']}} | "
            f"{'Shape':<{col_widths['shape']}} | "
            f"{'Avg (ms)':>{col_widths['avg_duration_ms']}} | "
            f"{'Min (ms)':>{col_widths['min_duration_ms']}} | "
            f"{'Max (ms)':>{col_widths['max_duration_ms']}}"
        )
        if has_cb_col:
            header += f" | {'CB Wait (%)':>{col_widths['cb_wait_pct']}}"
        separator = "-" * len(header)

        print()
        print(separator)
        print(header)
        print(separator)

        # Print rows
        for _, row in df.iterrows():
            # Handle signpost rows (with dashes)
            if row["avg_duration_ms"] == "-":
                line = (
                    f"{row['op_name']:<{col_widths['op_name']}} | "
                    f"{'-'*col_widths['shape']} | "
                    f"{'-'*col_widths['avg_duration_ms']:>{col_widths['avg_duration_ms']}} | "
                    f"{'-'*col_widths['min_duration_ms']:>{col_widths['min_duration_ms']}} | "
                    f"{'-'*col_widths['max_duration_ms']:>{col_widths['max_duration_ms']}}"
                )
                if has_cb_col:
                    line += f" | {'-':>{col_widths['cb_wait_pct']}}"
                print(line)
            else:
                line = (
                    f"{row['op_name']:<{col_widths['op_name']}} | "
                    f"{row['shape']:<{col_widths['shape']}} | "
                    f"{row['avg_duration_ms']:>{col_widths['avg_duration_ms']}} | "
                    f"{row['min_duration_ms']:>{col_widths['min_duration_ms']}} | "
                    f"{row['max_duration_ms']:>{col_widths['max_duration_ms']}}"
                )
                if has_cb_col:
                    line += f" | {str(row.get('cb_wait_pct', '-')):>{col_widths['cb_wait_pct']}}"
                print(line)

        print(separator)
        print()


def save_summary_csv(df: pd.DataFrame, output_path: str):
    """
    Save the summary DataFrame to a CSV file.

    Args:
        df: DataFrame with performance summary
        output_path: Path to save the CSV file
    """
    try:
        # Sort by config_name and impl_type if available (aggregated format)
        if "config_name" in df.columns and "impl_type" in df.columns:
            known = [t for t in IMPL_ORDER if t in df["impl_type"].values]
            unknown = sorted(t for t in df["impl_type"].unique() if t not in IMPL_ORDER)
            ordered_types = known + unknown
            df_sort = df.copy()
            df_sort["impl_type"] = pd.Categorical(df_sort["impl_type"], categories=ordered_types, ordered=True)
            df_sorted = df_sort.sort_values(["config_name", "impl_type"])
            df_sorted.to_csv(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"Summary saved to: {output_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Tracy profiler CSV outputs and generate performance summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze most recent Tracy CSV
  python analyze_tracy_perf.py

  # Analyze specific CSV file
  python analyze_tracy_perf.py --csv generated/profiler/reports/2026_01_27_14_32_26/ops_perf_results_2026_01_27_14_32_26.csv

  # Analyze most recent CSV in a specific directory
  python analyze_tracy_perf.py --dir generated/profiler/reports/2026_01_27_14_32_26/

  # Save output to custom CSV file
  python analyze_tracy_perf.py --output my_perf_summary.csv

  # Exclude signpost markers from output
  python analyze_tracy_perf.py --no-signposts

  # Run Tracy profiling and analyze results
  python analyze_tracy_perf.py --run "python test_script.py"
  python analyze_tracy_perf.py --run "pytest test_file.py::test_name"
  python analyze_tracy_perf.py --run "python test_dit_minimal_matmul_addcmul_fused_perf.py::test_benchmark_minimal_matmul_with_bias"

  # Run with full per-core data for compute vs. memory boundedness analysis
  python analyze_tracy_perf.py --run "python test_script.py" --no-runtime-analysis
        """,
    )

    parser.add_argument(
        "--run",
        type=str,
        help="Run command with Tracy profiling and analyze the results. Command will be prefixed with 'python -m tracy -r -p -v -m'",
    )

    parser.add_argument("--csv", type=str, help="Path to specific ops_perf_results CSV file to analyze")

    parser.add_argument("--dir", type=str, help="Directory to search for the most recent ops_perf_results CSV file")

    parser.add_argument("--output", "-o", type=str, help="Output CSV file path (default: perf_summary_<timestamp>.csv)")

    parser.add_argument(
        "--no-signposts",
        action="store_true",
        help="Exclude signpost markers from the output (only for non-aggregated mode)",
    )

    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results by implementation (1 row per impl, averages only, excludes warmup). Default when using --run.",
    )

    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Disable aggregation (show all iterations). Use with --run to see detailed output.",
    )

    parser.add_argument(
        "--no-runtime-analysis",
        action="store_true",
        default=False,
        help=(
            "Pass --no-runtime-analysis to tracy, disabling C++ post-processing of profiling data. "
            "This produces fuller per-core data (e.g. DEVICE COMPUTE CB WAIT FRONT) which is useful "
            "for determining whether ops are compute- or memory-bound. Off by default."
        ),
    )

    args = parser.parse_args()

    # Determine aggregation mode
    # Default: aggregate when using --run, unless --no-aggregate is specified
    use_aggregate = args.aggregate or (args.run and not args.no_aggregate)

    # Determine which CSV file to analyze
    csv_path = None

    if args.run:
        # Run Tracy command and get the generated CSV
        csv_path = run_tracy_command(args.run, args.dir, no_runtime_analysis=args.no_runtime_analysis)
        if csv_path is None:
            print("Error: Failed to run Tracy command or find generated CSV", file=sys.stderr)
            sys.exit(1)
        print(f"\nGenerated CSV: {csv_path}\n")
    elif args.csv:
        csv_path = args.csv
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
    else:
        csv_path = find_latest_tracy_csv(args.dir)
        if csv_path is None:
            sys.exit(1)

    print(f"Analyzing: {csv_path}")
    if use_aggregate:
        print("Mode: Aggregated (1 row per implementation, excluding warmup)")
    else:
        print("Mode: Detailed (all iterations)")
    print()

    # Analyze the CSV
    if use_aggregate:
        summary_df = analyze_tracy_csv_aggregated(csv_path)
    else:
        summary_df = analyze_tracy_csv(csv_path, include_signposts=not args.no_signposts)

    if summary_df.empty:
        print("No performance data found to analyze")
        sys.exit(1)

    # Print summary table
    print_summary_table(summary_df)

    # Save to CSV if output path specified or use default
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_path = f"perf_summary_{timestamp}.csv"

    save_summary_csv(summary_df, output_path)

    if use_aggregate:
        print(f"\nAnalyzed {len(summary_df)} implementation(s) from {csv_path}")
    else:
        print(f"\nAnalyzed {len(summary_df)} operations from {csv_path}")


if __name__ == "__main__":
    main()
