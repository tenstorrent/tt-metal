# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Performance utilities for DeepSeek V3 multi-chip testing.

Provides functions for device performance testing with multi-device row merging
to accurately measure performance on multi-chip topologies.
"""

import pandas as pd
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.tt_transformers.tests.test_utils import merge_device_rows


def run_model_device_perf_test_with_merge(
    command: str,
    expected_device_perf_ns_per_iteration: float,
    subdir: str,
    model_name: str,
    num_iterations: int = 1,
    batch_size: int = 1,
    margin: float = 0.015,
    comments: str = "",
):
    """
    Run device performance test with multi-device row merging.

    This function extends run_model_device_perf_test by adding device row merging
    for accurate multi-chip performance measurement. In multi-chip scenarios:
    - Collective operations (AllGather, ReduceScatter, AllReduce) use AVERAGE duration
    - Non-collective operations use MAX duration (critical path)

    Wraps the standard workflow from run_model_device_perf_test with an additional
    merge step between profiling and validation.

    Args:
        command: Command to execute for running the model
        expected_device_perf_ns_per_iteration: Expected device kernel duration in nanoseconds
        subdir: Subdirectory where performance logs will be stored
        model_name: Name of the model being tested
        num_iterations: Number of iterations (default: 1)
        batch_size: Batch size for the model (default: 1)
        margin: Acceptable performance margin as percentage (default: 0.015 = 1.5%)
        comments: Additional settings description for the report
    """
    # Same setup as run_model_device_perf_test
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"

    # Step 1: Run device profiling (same as run_model_device_perf_test)
    post_processed_results = run_device_perf(
        command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
    )

    # Step 2: Apply multi-device row merging (ADDITIONAL STEP for multi-chip)
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    # Validate and log signpost filtering
    total_rows = len(df)
    signpost_rows = len(df[df["OP TYPE"] == "tt_signpost"])
    device_rows = len(df[df["OP TYPE"] == "tt_dnn_device"])

    logger.debug(f"CSV total rows: {total_rows}, signposts: {signpost_rows}, device ops: {device_rows}")

    # Filter to device operations only
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    assert len(df) == device_rows, "Filtering failed: unexpected row count"
    assert len(df[df["OP TYPE"] == "tt_signpost"]) == 0, "Signposts not properly filtered out"

    # Merge device rows
    logger.debug(f"Device rows before merge: {len(df)}")
    df_merged = merge_device_rows(df)
    logger.debug(f"Device rows after merge: {len(df_merged)}")

    # Recalculate metrics from merged data
    if not df_merged.empty:
        merged_kernel_durations = df_merged["DEVICE KERNEL DURATION [ns]"].dropna().tolist()
        if merged_kernel_durations:
            merged_sum_ns = sum(merged_kernel_durations)
            logger.debug(f"Merged operations count: {len(merged_kernel_durations)}")
            logger.debug(f"Merged sum (ns): {merged_sum_ns} ({merged_sum_ns / 1000:.1f} μs)")
            logger.debug(f"Original {inference_time_key}: {post_processed_results.get(inference_time_key, 'N/A')}")
            # Update with merged value
            post_processed_results[inference_time_key] = merged_sum_ns

    # Steps 3-4: Validation and reporting (same as run_model_device_perf_test)
    expected_perf_cols = {inference_time_key: expected_device_perf_ns_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=margin, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=comments,
    )
