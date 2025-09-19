# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import os
import re
import subprocess
import pytest
from loguru import logger
from pathlib import Path

from tracy.process_device_log import import_log_run_stats
import tracy.device_post_proc_config as device_post_proc_config


# Helper function to sort strings naturally
def natural_key_from_path(s):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s.name)]


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
STATIC_ONLY_SHARDED_ARGS_CONFIGS = [
    "0000001",  # Everything static, sharded tensor
]

STATIC_ONLY_INTERLEAVED_ARGS_CONFIGS = [
    "0000000",  # Everything static, interleaved tensor
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


def impl_test(gtest_filter, res_dir, args_configs=None):
    if args_configs is None:
        args_configs = ARGS_CONFIGS

    ENV = os.environ.copy()
    ENV["TT_METAL_DEVICE_PROFILER"] = "1"
    ENV["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
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
    for test_dir in sorted(results_dir.iterdir(), key=natural_key_from_path):
        if not test_dir.is_dir():
            raise IsADirectoryError(f"Expected {test_dir} to be a directory containing {profile_log_file}")
        profile_log_path = test_dir / "profile_log_device.csv"
        setup.deviceInputLog = profile_log_path

        stats = import_log_run_stats(setup)
        logger.info(f"Results for {test_dir}:")
        for zone_name in zone_names:
            core = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"][0]
            st = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"][zone_name]["stats"]
            logger.info(f"Zone: {zone_name}: Average: {st['Average']} (cycles)")


def impl_test_sharded_static_only(gtest_filter, res_dir):
    """Implementation for tests that only run with all-static configuration."""
    impl_test(gtest_filter, res_dir, STATIC_ONLY_SHARDED_ARGS_CONFIGS)


def impl_test_interleaved_static_only(gtest_filter, res_dir):
    """Implementation for tests that only run with all-static configuration."""
    impl_test(gtest_filter, res_dir, STATIC_ONLY_INTERLEAVED_ARGS_CONFIGS)


def test_get_noc_addr_page_id():
    impl_test("AccessorTests/AccessorBenchmarks.GetNocAddr/*", res_dir="accessor_get_noc_addr_benchmarks")


def test_get_noc_addr_page_coord():
    impl_test(
        "AccessorTests/AccessorBenchmarks.GetNocAddrPageCoord/*", res_dir="accessor_get_noc_addr_page_coord_benchmarks"
    )


def test_constructor():
    impl_test("AccessorTests/AccessorBenchmarks.Constructor/*", res_dir="accessor_constructor_benchmarks")


def test_manual_pages_iteration_sharded():
    impl_test_sharded_static_only(
        "AccessorTests/AccessorBenchmarks.ManualPagesIterationSharded/*",
        res_dir="accessor_manual_pages_iteration_sharded_benchmarks",
    )


def test_pages_iterator_sharded():
    impl_test_sharded_static_only(
        "AccessorTests/AccessorBenchmarks.PagesIteratorSharded/*", res_dir="accessor_pages_iterator_sharded_benchmarks"
    )


def test_manual_pages_iteration_interleaved():
    impl_test_interleaved_static_only(
        "AccessorTests/AccessorBenchmarks.ManualPagesIterationInterleaved/*",
        res_dir="accessor_manual_pages_iteration_interleaved_benchmarks",
    )


def test_pages_iterator_interleaved():
    impl_test_interleaved_static_only(
        "AccessorTests/AccessorBenchmarks.PagesIteratorInterleaved/*",
        res_dir="accessor_pages_iterator_interleaved_benchmarks",
    )
