#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import csv
import sys
import yaml
from loguru import logger  # type: ignore
from matplotlib.gridspec import GridSpec
import itertools
import numpy as np
from collections import defaultdict


from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

from tests.tt_metal.tt_metal.data_movement.python.config import DataMovementConfig
from tests.tt_metal.tt_metal.data_movement.python.test_metadata import TestMetadataLoader
from tests.tt_metal.tt_metal.data_movement.python.stats_collector import StatsCollector
from tests.tt_metal.tt_metal.data_movement.python.stats_reporter import StatsReporter
from tests.tt_metal.tt_metal.data_movement.python.plotter import Plotter
from tests.tt_metal.tt_metal.data_movement.python.constants import *


def run_dm_tests(profile, verbose, gtest_filter, plot, report, arch_name):
    logger.info("Starting data movement tests...")
    log_file_path = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"

    # Initialize configuration and load test metadata
    config = DataMovementConfig()
    metadata_loader = TestMetadataLoader(config)
    test_id_to_name, test_id_to_comment, test_bounds, test_type_attributes = metadata_loader.get_test_mappings()

    # Get architecture
    arch = config.get_arch(arch_name, test_bounds)
    if verbose:
        logger.info(f"Running data movement tests on architecture: {arch}")

    # Profile tests
    if profile or not os.path.exists(log_file_path) or gtest_filter:
        profile_dm_tests(verbose=verbose, gtest_filter=gtest_filter)

    # Gather analysis stats
    stats_collector = StatsCollector(log_file_path, test_id_to_name, test_type_attributes, verbose=verbose)
    dm_stats, aggregate_stats = stats_collector.gather_analysis_stats()

    # Print stats if explicitly requested
    stats_reporter = StatsReporter(
        dm_stats, aggregate_stats, test_id_to_name, test_type_attributes, DEFAULT_OUTPUT_DIR, arch
    )

    if verbose:
        stats_reporter.print_stats()

    # Export results to csv
    if report:
        stats_reporter.export_dm_stats_to_csv()

    # Plot results
    if plot:
        plotter = Plotter(dm_stats, aggregate_stats, DEFAULT_OUTPUT_DIR, arch, test_id_to_name, test_id_to_comment)
        plotter.plot_dm_stats()

    # Check performance
    performance_check(dm_stats, arch=arch, verbose=verbose, test_bounds=test_bounds, test_id_to_name=test_id_to_name)

    logger.info("Data movement tests completed.")


def profile_dm_tests(verbose=False, gtest_filter=None):
    if verbose:
        logger.info(f"Profiling Kernels...")
    cmd = f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/unit_tests_data_movement"

    if gtest_filter:
        cmd += f' --gtest_filter="*{gtest_filter}*"'

    os.system(cmd)


def performance_check(dm_stats, arch="blackhole", verbose=False, test_bounds=None, test_id_to_name=None):
    # Tidy results' ranges
    results_bounds = {}
    for riscv in dm_stats.keys():
        for run in dm_stats[riscv]["analysis"]["series"]:
            run_host_id = run["duration_type"][0]["run_host_id"]
            test_id = dm_stats[riscv]["attributes"][run_host_id]["Test id"]

            if test_id not in results_bounds.keys():
                results_bounds[test_id] = {riscv: {"latency": 0, "bandwidth": float("inf")}}
            elif riscv not in results_bounds[test_id].keys():
                results_bounds[test_id][riscv] = {
                    "latency": 0,
                    "bandwidth": float("inf"),
                }

            cycles = run["duration_cycles"]
            results_bounds[test_id][riscv]["latency"] = max(results_bounds[test_id][riscv]["latency"], cycles)

            attributes = dm_stats[riscv]["attributes"][run_host_id]
            bandwidth = attributes["Number of transactions"] * attributes["Transaction size in bytes"] / cycles
            results_bounds[test_id][riscv]["bandwidth"] = min(results_bounds[test_id][riscv]["bandwidth"], bandwidth)

    # Performance checks per test
    for test_id, bounds in results_bounds.items():
        # Print latency and bandwidth perf results
        if verbose:
            logger.info("")
            test_name = test_id_to_name.get(test_id, "Unknown Test")
            logger.info(f"Perf results for test id: {test_id} ({test_name})")

            logger.info(f"Latency")
            for riscv in bounds.keys():
                if bounds[riscv]["latency"] != float("inf"):
                    logger.info(f"  {riscv}: {bounds[riscv]['latency']} cycles")

            logger.info(f"Bandwidth")
            for riscv in bounds.keys():
                if bounds[riscv]["bandwidth"] != float("inf"):
                    logger.info(f"  {riscv}: {bounds[riscv]['bandwidth']} Bytes/cycle")

        if test_bounds is None or test_id not in test_bounds[arch].keys():
            logger.warning(f"Test id {test_id} not found in {arch} test bounds.")
            continue

        for riscv in bounds.keys():
            if riscv not in test_bounds[arch][test_id].keys():
                continue

            cycles_within_bounds = bounds[riscv]["latency"] <= test_bounds[arch][test_id][riscv]["latency"]
            bw_within_bounds = test_bounds[arch][test_id][riscv]["bandwidth"] <= bounds[riscv]["bandwidth"]

            # Print bounds check results
            if verbose:
                if not cycles_within_bounds:
                    logger.warning(f"{riscv} cycles not within perf bounds.")
                else:
                    logger.info(f"{riscv} cycles within perf bounds.")
                if not bw_within_bounds:
                    logger.warning(f"{riscv} bandwidth not within perf bounds.")
                else:
                    logger.info(f"{riscv} bandwidth within perf bounds.")

            assert cycles_within_bounds
            assert bw_within_bounds


def test_data_movement(
    no_profile: bool,
    verbose_log: bool,
    gtest_filter: str,
    plot: bool,
    report: bool,
    arch: str,
):
    run_dm_tests(
        profile=not no_profile, verbose=verbose_log, gtest_filter=gtest_filter, plot=plot, report=report, arch_name=arch
    )
