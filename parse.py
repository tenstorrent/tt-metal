#!/usr/bin/env python3
"""Parse reduce scatter perf results from profiler CSV."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Hardcoded values
NUM_ITERS = 10
NUM_DEVICES = 8
# ROWS_PER_CONFIG = 3 * NUM_ITERS * NUM_DEVICES
ROWS_PER_CONFIG = (1 + 2 * NUM_ITERS) * NUM_DEVICES


def find_latest_csv(base_dir: Path) -> Path:
    """Find the latest CSV file in timestamped profiler reports directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Profiler reports directory not found: {base_dir}")

    # Find all timestamped directories (format: YYYY_MM_DD_HH_MM_SS)
    dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.replace("_", "").isdigit()],
        reverse=True,
    )
    if not dirs:
        raise FileNotFoundError(f"No timestamped directories found in {base_dir}")

    latest_dir = dirs[0]

    # Find CSV file matching pattern ops_perf_results_*.csv
    csvs = list(latest_dir.glob("ops_perf_results_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No ops_perf_results CSV found in {latest_dir}")

    return csvs[0]


def get_pytest_configs(test_path: str) -> list[str]:
    """Run pytest --collect-only and parse test config names."""
    try:
        result = subprocess.run(
            ["pytest", test_path, "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Parse test names from output
        configs = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and "<Function " in line:
                # Extract the parameterized test name part after ::test_rs_perf
                config_name = line.split("[", 1)[1].split("]", 1)[0]
                configs.append(config_name)
        return configs
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Warning: Could not run pytest collect-only: {e}", file=sys.stderr)
        return []


def compute_avg_kernel_duration(df_config: pd.DataFrame) -> float:
    """
    Compute average kernel duration for a test config.

    Takes last NUM_ITERS * NUM_DEVICES rows, groups by iteration,
    takes max across devices, then averages across iterations.
    """
    # Take last 1*num_iters*num_devices rows
    last_rows = df_config.tail(NUM_ITERS * NUM_DEVICES)

    if len(last_rows) != NUM_ITERS * NUM_DEVICES:
        raise ValueError(f"Expected {NUM_ITERS * NUM_DEVICES} rows, got {len(last_rows)}")

    kernel_durations = last_rows["DEVICE KERNEL DURATION [ns]"].values

    # Group into iterations and take max per iteration
    max_per_iter = []
    for i in range(NUM_ITERS):
        start_idx = i * NUM_DEVICES
        end_idx = start_idx + NUM_DEVICES
        iter_durations = kernel_durations[start_idx:end_idx]
        max_per_iter.append(max(iter_durations))

    # Average across iterations
    return sum(max_per_iter) / len(max_per_iter)


def main():
    parser = argparse.ArgumentParser(description="Parse reduce scatter perf results")
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Path to CSV file (optional, auto-discovers latest if not provided)",
    )
    parser.add_argument(
        "--test-path",
        # default="tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_rs_perf",
        default="tests/nightly/t3000/ccl/test_new_all_broadcast.py::test_ab_perf",
        help="Pytest test path for collecting configs",
    )
    args = parser.parse_args()

    # Find CSV file
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
    else:
        base_dir = Path("generated/profiler/reports")
        try:
            csv_path = find_latest_csv(base_dir)
            print(f"Using latest CSV: {csv_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Get pytest configs
    configs = get_pytest_configs(args.test_path)
    if not configs:
        print("Warning: Could not get pytest configs, using generic names", file=sys.stderr)

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    total_rows = len(df)
    num_configs = total_rows // ROWS_PER_CONFIG

    # Validate row count
    expected_rows = len(configs) * ROWS_PER_CONFIG
    if total_rows != expected_rows:
        print(
            f"Error: Row count mismatch. Got {total_rows}, expected {expected_rows} "
            f"({num_configs} configs * {ROWS_PER_CONFIG} rows/config)",
            file=sys.stderr,
        )
        return

    if num_configs == 0:
        print(
            f"Error: Not enough rows ({total_rows}) for even one config ({ROWS_PER_CONFIG} required)", file=sys.stderr
        )
        sys.exit(1)

    print(f"Total rows: {total_rows}, Configs: {num_configs}")
    print(f"Expected: {num_configs} * {ROWS_PER_CONFIG} = {num_configs * ROWS_PER_CONFIG}")
    print()

    # Process each config
    print(configs)
    results = []
    for i in range(num_configs):
        start_row = i * ROWS_PER_CONFIG
        end_row = start_row + ROWS_PER_CONFIG
        df_config = df.iloc[start_row:end_row]

        try:
            avg_duration = compute_avg_kernel_duration(df_config)
        except ValueError as e:
            print(f"Error processing config {i}: {e}", file=sys.stderr)
            continue

        results.append((configs[i], avg_duration))

    # Print results
    print("=" * 60)
    print("Avg Kernel Duration per Config")
    print("=" * 60)
    for config_name, avg_duration in results:
        print(f"{config_name}: {(avg_duration/1000):.2f} us")


if __name__ == "__main__":
    main()
