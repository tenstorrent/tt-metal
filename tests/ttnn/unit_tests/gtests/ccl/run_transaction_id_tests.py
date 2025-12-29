#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
import csv
from datetime import datetime
from pathlib import Path
import pytest
from loguru import logger
import pandas as pd
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

# Define profiler log paths based on the example
PROFILER_LOGS_DIR = Path(os.environ.get("TT_METAL_HOME", ".")) / "generated" / "profiler" / ".logs"
PROFILER_DEVICE_SIDE_LOG = "profile_log_device.csv"
profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

# Define the results CSV file
RESULTS_CSV = "transaction_id_throughput_results.csv"


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def calculate_throughput(data_size_bytes, cycles, freq_mhz):
    """Calculate throughput in GB/s from cycles and data size"""
    # Convert to seconds
    time_seconds = cycles / (freq_mhz * 1e6)
    # Calculate throughput in GB/s
    throughput_gb_per_s = (data_size_bytes / 1e9) / time_seconds
    return throughput_gb_per_s


def profile_results(zone_name, page_size, num_pages):
    freq = get_device_freq() / 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    main_test_body_string = zone_name
    setup.timerAnalysis = {
        main_test_body_string: {
            "across": "device",
            "type": "session_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": main_test_body_string},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": main_test_body_string},
        },
    }
    devices_data = import_log_run_stats(setup)
    devices = list(devices_data["devices"].keys())

    # MAIN-TEST-BODY
    main_loop_cycles = []
    for device in devices:
        main_loop_cycle = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string][
            "stats"
        ]["Average"]
        main_loop_cycles.append(main_loop_cycle)

    total_byte_sent = page_size * num_pages
    bandwidth = total_byte_sent / max(main_loop_cycles)

    return bandwidth


def extract_profiling_data(page_size, num_pages):
    writer_bw_GB_per_second = profile_results("MAIN-LOOP", page_size, num_pages)
    reader_bw_GB_per_second = profile_results("NCRISC-KERNEL", page_size, num_pages)

    return writer_bw_GB_per_second, reader_bw_GB_per_second


def run_transaction_id_test(
    page_size,
    num_pages,
    is_dram,
    do_write_barrier,
    num_reader_transaction_ids,
    num_writer_transaction_ids,
    expected_throughput=None,
    do_correctness_check=False,
):
    """Run the transaction ID tracker test with given parameters"""
    os.system(f"rm -f {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")
    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/ttnn/unit_tests_ttnn_transaction_id_tracker \
            {page_size} \
            {num_pages} \
            {int(is_dram)} \
            {int(do_write_barrier)} \
            {num_reader_transaction_ids} \
            {num_writer_transaction_ids} \
            {int(do_correctness_check)}"

    # Clear previous profiling data
    if os.path.exists(profiler_log_path):
        try:
            os.remove(profiler_log_path)
            logger.info(f"Removed previous profiler log at {profiler_log_path}")
        except:
            logger.warning(f"Could not remove previous profiler log at {profiler_log_path}")

    logger.info(
        f"Running test with: page_size={page_size}, num_pages={num_pages}, "
        f"is_dram={is_dram}, do_write_barrier={do_write_barrier}, "
        f"num_reader_transaction_ids={num_reader_transaction_ids}, num_writer_transaction_ids={num_writer_transaction_ids}, do_correctness_check={do_correctness_check}"
    )

    # Set environment variable for profiling
    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"

    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False


# Initialize results CSV before running tests
@pytest.fixture(scope="session", autouse=True)
def init_results_file():
    """Initialize the results CSV file before any tests run"""
    # Delete the results file if it exists
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
        logger.info(f"Removed previous results file: {RESULTS_CSV}")

    # Create a new file with headers
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        fieldnames = [
            "page_size",
            "num_pages",
            "is_dram",
            "do_write_barrier",
            "num_reader_transaction_ids",
            "num_writer_transaction_ids",
            "write_throughput_GB_s",
            "read_throughput_GB_s",
            "test_type",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    logger.info(f"Created new results file: {RESULTS_CSV}")

    # Run all tests
    yield

    # After all tests have completed, generate summary report
    generate_summary_report()


def append_to_results_csv(test_results):
    """Append test results to the CSV file"""
    with open(RESULTS_CSV, "a", newline="") as csvfile:
        fieldnames = [
            "page_size",
            "num_pages",
            "is_dram",
            "do_write_barrier",
            "num_reader_transaction_ids",
            "num_writer_transaction_ids",
            "write_throughput_GB_s",
            "read_throughput_GB_s",
            "test_type",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(test_results)


def generate_summary_report():
    """Generate a summary report from the results CSV"""
    if not os.path.exists(RESULTS_CSV):
        logger.warning("No results file found, cannot generate summary.")
        return

    try:
        df = pd.read_csv(RESULTS_CSV)

        if len(df) == 0:
            logger.warning("Results file is empty, cannot generate summary.")
            return

        logger.info("\n" + "=" * 70)
        logger.info("TRANSACTION ID THROUGHPUT TEST SUMMARY REPORT")
        logger.info("=" * 70)

        # Overall stats
        logger.info("\nOverall Statistics:")
        logger.info(f"Total tests run: {len(df)}")
        logger.info(f"Average write throughput: {df['write_throughput_GB_s'].mean():.2f} GB/s")
        logger.info(f"Max write throughput: {df['write_throughput_GB_s'].max():.2f} GB/s")
        logger.info(f"Average read throughput: {df['read_throughput_GB_s'].mean():.2f} GB/s")
        logger.info(f"Max read throughput: {df['read_throughput_GB_s'].max():.2f} GB/s")

        # Raw results - list all test cases
        logger.info("\nRaw Test Results:")
        logger.info("-" * 120)
        logger.info(
            f"{'Page Size':>10} {'Pages':>10} {'Memory':>10} {'Barrier':>10} {'Reader TIDs':>12} {'Writer TIDs':>12} {'Write BW (GB/s)':>15} {'Read BW (GB/s)':>15}"
        )
        logger.info("-" * 120)

        # Sort by page size, is_dram, do_write_barrier, and num_transaction_ids for better readability
        sorted_df = df.sort_values(by=["page_size", "is_dram", "do_write_barrier", "num_writer_transaction_ids"])

        for _, row in sorted_df.iterrows():
            logger.info(
                f"{row['page_size']:>10} {row['num_pages']:>10} {'DRAM' if row['is_dram'] else 'L1':>10} "
                f"{'Yes' if row['do_write_barrier'] else 'No':>10} {row['num_reader_transaction_ids']:>12} "
                f"{row['num_writer_transaction_ids']:>12} {row['write_throughput_GB_s']:>15.2f} "
                f"{row['read_throughput_GB_s']:>15.2f}"
            )

        # Detailed analysis by page size and write barrier mode
        logger.info("\n" + "=" * 70)
        logger.info("DETAILED PERFORMANCE BY PAGE SIZE, BARRIER MODE, AND TRANSACTION IDS")
        logger.info("=" * 70)

        # Get unique page sizes and barrier modes
        page_sizes = sorted(df["page_size"].unique())
        barrier_modes = [False, True]

        for page_size in page_sizes:
            for barrier_mode in barrier_modes:
                logger.info(
                    f"\nPage Size: {page_size} bytes, Write Barrier: {'Enabled' if barrier_mode else 'Disabled'}"
                )
                logger.info("-" * 70)

                # Filter data for this page size and barrier mode
                subset = df[(df["page_size"] == page_size) & (df["do_write_barrier"] == barrier_mode)]

                # Group by memory type and transaction ID count
                for is_dram in [False, True]:
                    memory_type = "DRAM" if is_dram else "L1"
                    mem_subset = subset[subset["is_dram"] == is_dram]

                    if len(mem_subset) > 0:
                        logger.info(f"\n  Memory Type: {memory_type}")
                        logger.info(f"  {'-' * 60}")
                        logger.info(f"  {'Writer TIDs':>12} {'Write BW (GB/s)':>15} {'Read BW (GB/s)':>15}")
                        logger.info(f"  {'-' * 50}")

                        # Sort by transaction ID count
                        mem_subset = mem_subset.sort_values("num_writer_transaction_ids")

                        for _, row in mem_subset.iterrows():
                            logger.info(
                                f"  {row['num_writer_transaction_ids']:>12} {row['write_throughput_GB_s']:>15.2f} "
                                f"{row['read_throughput_GB_s']:>15.2f}"
                            )

        # By page size
        logger.info("\nThroughput by Page Size (GB/s):")
        logger.info("-" * 40)
        page_size_summary = df.groupby("page_size").agg(
            {"write_throughput_GB_s": ["mean", "max"], "read_throughput_GB_s": ["mean", "max"]}
        )
        logger.info(f"\n{page_size_summary}")

        # By memory type
        logger.info("\nThroughput by Memory Type (GB/s):")
        logger.info("-" * 40)
        memory_type_df = df.copy()
        memory_type_df["memory_type"] = memory_type_df["is_dram"].map({True: "DRAM", False: "L1"})
        memory_summary = memory_type_df.groupby("memory_type").agg(
            {"write_throughput_GB_s": ["mean", "max"], "read_throughput_GB_s": ["mean", "max"]}
        )
        logger.info(f"\n{memory_summary}")

        logger.info("=" * 70)
        logger.info(f"Detailed results saved to {RESULTS_CSV}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error generating summary report: {e}")


# Define test parameters for pytest parameterization
@pytest.mark.parametrize("page_size", [512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("num_pages", [2048])
@pytest.mark.parametrize("is_dram", [False, True])
@pytest.mark.parametrize("do_write_barrier", [False, True])
@pytest.mark.parametrize("num_writer_transaction_ids", [0, 1, 2, 4, 8, 16])
def test_transaction_id_throughput(page_size, num_pages, is_dram, do_write_barrier, num_writer_transaction_ids):
    """Parametrized test for transaction ID throughput measurement"""
    # Run the test and get throughput results
    do_correctness_check = False
    num_reader_transaction_ids = 0  # to disable reading
    run_transaction_id_test(
        page_size,
        num_pages,
        is_dram,
        do_write_barrier,
        num_reader_transaction_ids,
        num_writer_transaction_ids,
        do_correctness_check=do_correctness_check,
    )

    # Extract profiling data
    write_BW, read_BW = extract_profiling_data(page_size, num_pages)

    # Ensure throughput was measured
    if write_BW is not None:
        assert write_BW > 0, "Write throughput should be positive"

    if read_BW is not None:
        assert read_BW > 0, "Read throughput should be positive"

    # Create test result dictionary
    test_result = {
        "page_size": page_size,
        "num_pages": num_pages,
        "is_dram": is_dram,
        "do_write_barrier": do_write_barrier,
        "num_reader_transaction_ids": num_reader_transaction_ids,
        "num_writer_transaction_ids": num_writer_transaction_ids,
        "write_throughput_GB_s": write_BW,
        "read_throughput_GB_s": read_BW,
        "test_type": "throughput",
    }

    # Append to the results CSV
    append_to_results_csv(test_result)

    # Print individual test result
    logger.info("\n" + "=" * 50)
    logger.info("Test Configuration and Results")
    logger.info("=" * 50)
    logger.info(f"Page Size: {page_size} bytes")
    logger.info(f"Number of Pages: {num_pages}")
    logger.info(f"Memory Type: {'DRAM' if is_dram else 'L1'}")
    logger.info(f"Write Barrier: {'Yes' if do_write_barrier else 'No'}")
    logger.info(f"Writer Transaction IDs: {num_writer_transaction_ids}")
    logger.info(f"Write Throughput: {write_BW:.2f} GB/s")
    logger.info(f"Read Throughput: {read_BW:.2f} GB/s")
    logger.info("=" * 50)


@pytest.mark.parametrize("page_size", [512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("num_pages", [2048])
@pytest.mark.parametrize("is_dram", [True])
@pytest.mark.parametrize("do_write_barrier", [False, True])
@pytest.mark.parametrize("num_writer_transaction_ids", [0, 1, 2, 4, 8])
def test_transaction_id_correctness(page_size, num_pages, is_dram, do_write_barrier, num_writer_transaction_ids):
    """Parametrized test for transaction ID correctness verification"""
    do_correctness_check = True
    num_reader_transaction_ids = 0

    run_transaction_id_test(
        page_size,
        num_pages,
        is_dram,
        do_write_barrier,
        num_reader_transaction_ids,
        num_writer_transaction_ids,
        do_correctness_check=do_correctness_check,
    )
