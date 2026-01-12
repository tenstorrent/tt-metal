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


from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG, clear_profiler_runtime_artifacts

from tests.tt_metal.tt_metal.data_movement.python.config import DataMovementConfig
from tests.tt_metal.tt_metal.data_movement.python.test_metadata import TestMetadataLoader
from tests.tt_metal.tt_metal.data_movement.python.stats_collector import StatsCollector
from tests.tt_metal.tt_metal.data_movement.python.stats_reporter import StatsReporter
from tests.tt_metal.tt_metal.data_movement.python.plotter import Plotter
from tests.tt_metal.tt_metal.data_movement.python.performance_checker import PerformanceChecker
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
    stats = stats_collector.gather_stats_from_csv()
    if not stats.get("devices"):
        logger.info("No profiling data available.")
        return
    dm_stats, aggregate_stats = stats_collector.gather_analysis_stats(stats)

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
    performance_checker = PerformanceChecker(
        dm_stats, arch=arch, verbose=verbose, test_bounds=test_bounds, test_id_to_name=test_id_to_name
    )
    performance_checker.run()

    logger.info("Data movement tests completed.")


def profile_dm_tests(verbose=False, gtest_filter=None):
    if verbose:
        logger.info(f"Profiling Kernels...")

    clear_profiler_runtime_artifacts()

    cmd = f"TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=1333 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/unit_tests_data_movement"

    if gtest_filter:
        cmd += f' --gtest_filter="*{gtest_filter}*"'

    os.system(cmd)


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
