# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Accessor benchmarks.

To run benchmark and save results to json file (to generate ground truth data):
python accessor_benchmarks.py --export-results-to ./results

More details: python3 accessor_benchmarks.py --help
"""


import argparse
import json
import os
import re
import subprocess
import sys
from loguru import logger
from pathlib import Path

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
import numpy as np


# Helper function to sort strings naturally
def natural_key_from_path(s):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s.name)]


# TODO: Maybe use orjson?
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


"""
Runs tests/ttnn/unit_tests/gtests/accessor/test_accessor_benchmarks.cpp test and parses output.
See the C++ test for more details on what is benchmarked and generated.

Make sure that tt-metal is built with profiler enabled, e.g.:
./build_metal.sh --release --enable-ccache --build-tests -e -p
"""

# Bits Encoding: [Bank coordinates][Tensor shape][Shard shape][Number of banks][Rank]01
# 1 means dynamic, 0 means static
# Last 01 means that data is in l1 (not dram) and sharded
ARGS_CONFIGS = [
    "0000001",  # Everything static
    "0010001",  # dynamic tensor shape
    "0100001",  # dynamic shard shape
    "0110001",  # dynamic tensor shape and shard shape
    "0110101",  # static number of banks and banks coordinates
    "1000001",  # dynamic bank coordinates
    "1001001",  # dynamic bank coordinates and number of banks
    "1010001",  # dynamic bank coordinates and tensor shape
    "1011001",  # static rank and shard shape
    "1100001",  # dynamic bank coordinates and shard shape
    "1101001",  # static rank and tensor shape
    "1110001",  # static rank and number of banks
    "1110101",  # static number of banks
    "1111001",  # static rank
    "1111101",  # Everything dynamic
]

# For benchmarks that only support all-static configuration
STATIC_ONLY_ARGS_CONFIGS = [
    "0000001",  # Everything static
]


def get_individual_test_names(gtest_filter):
    """
    Get the list of individual test names that match the given filter pattern.
    This resolves star patterns like AccessorTests/AccessorBenchmarks.GetNocAddr/*
    into individual test names like AccessorTests/AccessorBenchmarks.GetNocAddr/0, etc.
    """
    ENV = os.environ.copy()
    BASE = Path(ENV["TT_METAL_HOME"])
    binary_path = Path(BASE / "build" / "test" / "ttnn" / "unit_tests_ttnn_accessor")

    # Use --gtest_list_tests with the filter to get all matching test names
    result = subprocess.run(
        [binary_path, f"--gtest_filter={gtest_filter}", "--gtest_list_tests"], env=ENV, capture_output=True, text=True
    )

    if result.returncode != 0:
        logger.error(f"Failed to list tests: {result.stderr}")
        return []

    # Parse the output to extract individual test names
    test_names = []
    current_test_case = None

    for line in result.stdout.split("\n"):
        # Skip empty lines and the "Running main() from gmock_main.cc" line
        if not line.strip() or "Running main() from" in line:
            continue

        # Test case names don't have leading spaces and end with '.'
        if not line.startswith(" ") and line.endswith("."):
            current_test_case = line.rstrip(".")  # Remove trailing '.'
        # Test names have leading spaces
        elif line.startswith(" ") and current_test_case:
            # Extract just the test name part (before any comment)
            test_name = line.strip().split()[0]  # Take first word, ignore comments after #
            if test_name:
                full_test_name = f"{current_test_case}.{test_name}"
                test_names.append(full_test_name)

    return test_names


def benchmark_impl(gtest_filter, res_dir, export_results_to=None, args_configs=None, n_repeat=1):
    if args_configs is None:
        args_configs = ARGS_CONFIGS

    ENV = os.environ.copy()
    ENV["TT_METAL_DEVICE_PROFILER"] = "1"
    BASE = Path(ENV["TT_METAL_HOME"])

    binary_path = Path(BASE / "build" / "test" / "ttnn" / "unit_tests_ttnn_accessor")

    # Get individual test names if using star pattern
    if "*" in gtest_filter:
        individual_tests = get_individual_test_names(gtest_filter)
        logger.info(f"Found {len(individual_tests)} individual tests for pattern '{gtest_filter}'")

        # Run each test individually
        for test_name in individual_tests:
            logger.info(f"Running individual test: {test_name}")
            subprocess.run([binary_path, f"--gtest_filter={test_name}"], env=ENV)
    else:
        # Run the test as before for non-star patterns
        subprocess.run([binary_path, f"--gtest_filter={gtest_filter}"], env=ENV)

    setup = device_post_proc_config.default_setup()
    zone_names = []
    timerAnalysis = {}
    for args_config in args_configs:
        zone_name = f"SHARDED_ACCESSOR_{args_config}"
        timerAnalysis[zone_name] = {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": zone_name},
            "end": {"risc": "BRISC", "zone_name": zone_name},
        }
        zone_names.append(zone_name)
    setup.timerAnalysis = timerAnalysis

    results_dir = Path(BASE / res_dir)
    profile_log_file = Path(".logs/profile_log_device.csv")

    # Collect results for JSON export if requested
    benchmark_results = {}

    for test_dir in sorted(results_dir.iterdir(), key=natural_key_from_path):
        if not test_dir.is_dir():
            raise IsADirectoryError(f"Expected {test_dir} to be a directory containing {profile_log_file}")
        profile_log_path = test_dir / "profile_log_device.csv"
        setup.deviceInputLog = profile_log_path

        stats = import_log_run_stats(setup)
        logger.info(f"Results for {test_dir}:")

        # Extract rank from directory name (e.g., "rank_2" -> "rank_2")
        rank_key = test_dir.name
        benchmark_results[rank_key] = {}

        for zone_name in zone_names:
            core = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"][0]
            st = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"][zone_name]["stats"]
            logger.info(f"Zone: {zone_name}: Average: {st['Average'] / n_repeat:.2f} (cycles)")

            # Store results for JSON export
            benchmark_results[rank_key][zone_name] = list(np.array(st["Samples"]) / n_repeat)

    # Export results if requested
    if export_results_to is not None:
        export_benchmark_results(
            benchmark_results, export_results_to, res_dir.replace("accessor_", "").replace("_benchmarks", "")
        )

    return benchmark_results


def benchmark_impl_static_only(gtest_filter, res_dir, export_results_to=None, n_repeat=1):
    return benchmark_impl(
        gtest_filter, res_dir, export_results_to, args_configs=STATIC_ONLY_ARGS_CONFIGS, n_repeat=n_repeat
    )


def export_benchmark_results(benchmark_results, export_dir, benchmark_name):
    """Export benchmark results to JSON file."""
    export_path = Path(export_dir)

    # Create export directory if it doesn't exist
    export_path.mkdir(parents=True, exist_ok=True)

    # Create filename based on benchmark name
    json_filename = f"{benchmark_name}.json"
    output_file = export_path / json_filename

    # Write JSON output
    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Results exported to: {output_file}")


# Benchmark tensor_accessor.get_noc_addr(page_id)
def benchmark_get_noc_addr_page_id(export_results_to=None):
    return benchmark_impl(
        "AccessorTests/AccessorBenchmarks.GetNocAddr/*",
        res_dir="accessor_get_noc_addr_benchmarks",
        export_results_to=export_results_to,
        n_repeat=100,
    )


# Benchmark tensor_accessor.get_noc_addr(page_coord)
def benchmark_get_noc_addr_page_coord(export_results_to=None):
    return benchmark_impl(
        "AccessorTests/AccessorBenchmarks.GetNocAddrPageCoord/*",
        res_dir="accessor_get_noc_addr_page_coord_benchmarks",
        export_results_to=export_results_to,
        n_repeat=100,
    )


# Benchmark tensor_accessor constructor
def benchmark_constructor(export_results_to=None):
    return benchmark_impl(
        "AccessorTests/AccessorBenchmarks.Constructor/*",
        res_dir="accessor_constructor_benchmarks",
        export_results_to=export_results_to,
        n_repeat=100,
    )


# Benchmark how many cycles it takes to iterate over all pages using tensor_accessor.get_noc_addr(page_id)
def benchmark_manual_pages_iteration(export_results_to=None):
    return benchmark_impl_static_only(
        "AccessorTests/AccessorBenchmarks.ManualPagesIteration/*",
        res_dir="accessor_manual_pages_iteration_benchmarks",
        export_results_to=export_results_to,
        n_repeat=1,
    )


# Benchmark how many cycles it takes to iterate over all pages using tensor_accessor.pages()
def benchmark_pages_iterator(export_results_to=None):
    return benchmark_impl_static_only(
        "AccessorTests/AccessorBenchmarks.PagesIterator/*",
        res_dir="accessor_pages_iterator_benchmarks",
        export_results_to=export_results_to,
        n_repeat=1,
    )


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run accessor benchmarks with optional result export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python accessor_benchmarks.py
  python accessor_benchmarks.py --benchmark-name get_noc_addr_page_id
  python accessor_benchmarks.py --export-results-to ./results
  python accessor_benchmarks.py --benchmark-name test_constructor --export-results-to ./results
        """,
    )

    parser.add_argument(
        "--benchmark-name",
        choices=[
            "get_noc_addr_page_id",
            "get_noc_addr_page_coord",
            "constructor",
            "manual_pages_iteration",
            "pages_iterator",
        ],
        help="Choose which test to run (default: run all tests)",
    )

    parser.add_argument(
        "--export-results-to", type=str, help="Directory to export results to (creates if it doesn't exist)"
    )

    args = parser.parse_args()

    # Validate export directory if specified
    if args.export_results_to:
        export_path = Path(args.export_results_to)
        if export_path.exists() and not export_path.is_dir():
            logger.error(f"Error: --export-results-to path exists but is not a directory: {args.export_results_to}")
            sys.exit(1)

    # Map of benchmark names to functions
    benchmark_functions = {
        "get_noc_addr_page_id": benchmark_get_noc_addr_page_id,
        "get_noc_addr_page_coord": benchmark_get_noc_addr_page_coord,
        "constructor": benchmark_constructor,
        "manual_pages_iteration": benchmark_manual_pages_iteration,
        "pages_iterator": benchmark_pages_iterator,
    }

    if args.benchmark_name:
        # Run specific benchmark
        logger.info(f"Running benchmark: {args.benchmark_name}")
        benchmark_functions[args.benchmark_name](export_results_to=args.export_results_to)
    else:
        # Run all benchmarks
        logger.info("Running all benchmarks")
        for name, func in benchmark_functions.items():
            logger.info(f"Running benchmark: {name}")
            func(export_results_to=args.export_results_to)


if __name__ == "__main__":
    main()
