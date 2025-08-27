# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance regression tests for accessor benchmarks.

This module tests for performance regressions by comparing current benchmark results
against ground truth baselines using the generic performance regression framework.
"""

import os
import sys
from pathlib import Path

import pytest
from loguru import logger

# Add parent directory to path for importing accessor_benchmarks
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from accessor_benchmarks import benchmark_get_noc_addr_page_id, benchmark_get_noc_addr_page_coord, benchmark_constructor

from perf_regression import PerformanceData, check_regression, summarize_regression_results


def load_baseline(benchmark_name: str) -> PerformanceData:
    gt_file_map = {
        "get_noc_addr_page_id": "get_noc_addr.json",
        "get_noc_addr_page_coord": "get_noc_addr_page_coord.json",
        "constructor": "constructor.json",
    }

    if benchmark_name not in gt_file_map:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if not tt_metal_home:
        raise RuntimeError("TT_METAL_HOME environment variable not set")

    gt_path = (
        Path(tt_metal_home)
        / "tests"
        / "ttnn"
        / "benchmark"
        / "python"
        / "perf_regression"
        / "gt"
        / "accessor_benchmarks"
        / gt_file_map[benchmark_name]
    )

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    return PerformanceData.from_json_file(gt_path)


def run_current_benchmark(benchmark_name: str) -> PerformanceData:
    benchmark_functions = {
        "get_noc_addr_page_id": benchmark_get_noc_addr_page_id,
        "get_noc_addr_page_coord": benchmark_get_noc_addr_page_coord,
        "constructor": benchmark_constructor,
    }

    if benchmark_name not in benchmark_functions:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    results = benchmark_functions[benchmark_name](export_results_to=None)

    # Convert results to PerformanceData format
    # The accessor benchmarks return: {rank: {config: [samples]}}
    return PerformanceData.from_dict(results)


def run_benchmark_regression_test(benchmark_name: str):
    """Generic function to run regression test for any benchmark."""
    logger.info(f"Running performance regression test for {benchmark_name}")

    # Load baseline data
    try:
        baseline = load_baseline(benchmark_name)
    except Exception as e:
        pytest.skip(f"Could not load baseline for {benchmark_name}: {e}")

    # Run current benchmark
    try:
        current = run_current_benchmark(benchmark_name)
    except Exception as e:
        pytest.skip(f"Could not run benchmark {benchmark_name}: {e}")

    # Run regression check
    try:
        results = check_regression(baseline, current)
    except Exception as e:
        pytest.skip(f"Could not compare results for {benchmark_name}: {e}")

    # Summarize results
    summary = summarize_regression_results(results)

    # Log summary
    logger.info(f"Regression test summary for {benchmark_name}:")
    logger.info(f"  Total tests: {summary['total_tests']}")
    logger.info(f"  Regressions: {summary['regressions']}")
    logger.info(f"  Improvements: {summary['improvements']}")
    logger.info(f"  No change: {summary['no_change']}")

    # Log detailed results
    for section, subsections in results.items():
        for subsection, result in subsections.items():
            logger.info(f"  {section}.{subsection}: {result['message']}")

    # Fail if any regressions detected
    if summary["regressions"] > 0:
        regression_summary = "\\n".join(summary["regression_messages"])
        assert False, f"Performance regressions detected in {benchmark_name}:\\n{regression_summary}"


def test_constructor():
    """Test for performance regressions in constructor benchmark."""
    run_benchmark_regression_test("constructor")


def test_get_noc_addr_page_id():
    """Test for performance regressions in get_noc_addr_page_id benchmark."""
    run_benchmark_regression_test("get_noc_addr_page_id")


def test_get_noc_addr_page_coord():
    """Test for performance regressions in get_noc_addr_page_coord benchmark."""
    run_benchmark_regression_test("get_noc_addr_page_coord")
