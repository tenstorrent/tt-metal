# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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

    Extends run_model_device_perf_test by adding device row merging for accurate
    multi-chip performance measurement. In multi-chip scenarios:
    - Collective operations (AllGather, ReduceScatter, AllReduce) use AVERAGE duration
    - Non-collective operations use MAX duration (critical path)

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
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"

    post_processed_results = run_device_perf(
        command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
    )

    # Apply multi-device row merging
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    total_rows = len(df)
    signpost_rows = len(df[df["OP TYPE"] == "tt_signpost"])
    device_rows = len(df[df["OP TYPE"] == "tt_dnn_device"])

    logger.debug(f"CSV total rows: {total_rows}, signposts: {signpost_rows}, device ops: {device_rows}")

    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]

    logger.debug(f"Device rows before merge: {len(df)}")
    df_merged = merge_device_rows(df)
    logger.debug(f"Device rows after merge: {len(df_merged)}")

    if not df_merged.empty:
        merged_kernel_durations = df_merged["DEVICE KERNEL DURATION [ns]"].dropna().tolist()
        if merged_kernel_durations:
            merged_sum_ns = sum(merged_kernel_durations)
            logger.debug(f"Merged operations count: {len(merged_kernel_durations)}")
            logger.debug(f"Merged sum (ns): {merged_sum_ns} ({merged_sum_ns / 1000:.1f} us)")
            logger.debug(f"Original {inference_time_key}: {post_processed_results.get(inference_time_key, 'N/A')}")
            post_processed_results[inference_time_key] = merged_sum_ns

        durations = df_merged["DEVICE KERNEL DURATION [ns]"].fillna(0)
        op_codes = df_merged["OP CODE"].astype(str)
        is_matmul = op_codes.str.contains("Matmul", case=False, na=False)
        is_ccl = op_codes.str.contains("AllGather|ReduceScatter|AllReduce", na=False)
        is_sdpa = op_codes.str.contains("SDPA|ScaledDotProductAttention", na=False)
        is_other = ~(is_matmul | is_ccl | is_sdpa)

        matmul_ns = durations[is_matmul].sum()
        ccl_ns = durations[is_ccl].sum()
        sdpa_ns = durations[is_sdpa].sum()
        other_ns = durations[is_other].sum()
        total_ns = matmul_ns + ccl_ns + sdpa_ns + other_ns

        logger.info(f"Matmul time: {matmul_ns:>15,.0f} ns ({matmul_ns / 1e3:>10,.1f} us)")
        logger.info(f"CCL    time: {ccl_ns:>15,.0f} ns ({ccl_ns / 1e3:>10,.1f} us)")
        logger.info(f"SDPA   time: {sdpa_ns:>15,.0f} ns ({sdpa_ns / 1e3:>10,.1f} us)")
        logger.info(f"Other  time: {other_ns:>15,.0f} ns ({other_ns / 1e3:>10,.1f} us)")
        logger.info(f"Total  time: {total_ns:>15,.0f} ns ({total_ns / 1e3:>10,.1f} us)")

        other_breakdown = (
            df_merged.loc[is_other]
            .groupby(op_codes[is_other])["DEVICE KERNEL DURATION [ns]"]
            .sum()
            .sort_values(ascending=False)
        )
        if not other_breakdown.empty:
            logger.info("Other ops breakdown:")
            for op_code, dur_ns in other_breakdown.items():
                logger.info(f"  {op_code:<40} {dur_ns:>15,.0f} ns ({dur_ns / 1e3:>10,.1f} us)")

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
