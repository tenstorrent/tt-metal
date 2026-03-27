# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for DeepSeek V3 MoE (Mixture of Experts) layer.

This test suite measures device kernel performance for the full MoE pipeline
on mesh-8x4-2 topology (8 devices in linear configuration with 2 links).

The test executes the existing PCC test and measures performance using the
Tracy profiler. Initial performance targets are dummy values that should be
updated after the first run with actual measured values.
"""

import pytest
import pandas as pd
import math
from loguru import logger
from collections import defaultdict
from tracy.process_model_log import get_latest_ops_log_filename
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


def merge_device_rows(df):
    """
    Merge multi-device operation rows into single rows.

    For collective operations (AllGather, ReduceScatter, AllReduce, Matmul_RS):
      Uses AVERAGE duration across devices (synchronized operations)

    For non-collective operations:
      Uses MAX duration across devices (critical path bottleneck)

    Args:
        df: pandas DataFrame with profiler data

    Returns:
        DataFrame with merged rows
    """
    block_by_device = defaultdict(list)

    # Group operations by device
    for _, row in df.iterrows():
        op_name = row["OP CODE"]
        op_type = row["OP TYPE"]
        if op_type == "tt_dnn_device":
            device_id = int(row["DEVICE ID"])
            block_by_device[device_id].append((op_name, row.to_dict()))

    device_ids = sorted(block_by_device.keys())
    merged_blocks = []

    # Synchronously pop operations from each device
    while device_ids and max(len(block_by_device[device_id]) for device_id in device_ids) > 0:
        blocks = []
        op_name = None

        for device_id in device_ids:
            if len(block_by_device[device_id]) > 0:
                current_op_name, current_block = block_by_device[device_id].pop(0)
                if op_name is None:
                    op_name = current_op_name
                elif op_name != current_op_name:
                    logger.warning(f"Mismatched operations: {op_name} vs {current_op_name}")
                blocks.append((device_id, current_block))
            else:
                logger.warning(f"Device {device_id} missing operation at this index")

        if not blocks:
            continue

        # Determine merging strategy based on operation type
        is_collective = (
            "AllGather" in op_name
            or "ReduceScatter" in op_name
            or "AllReduce" in op_name
            or "Matmul_RS" in op_name
        )

        if is_collective:
            # Collective ops: use AVERAGE duration
            device_kernel_durations = [
                d["DEVICE KERNEL DURATION [ns]"]
                for _, d in blocks
                if "DEVICE KERNEL DURATION [ns]" in d and not math.isnan(d["DEVICE KERNEL DURATION [ns]"])
            ]
            average_duration = (
                sum(device_kernel_durations) / len(device_kernel_durations) if device_kernel_durations else float("nan")
            )
            base_block = blocks[0][1].copy()
            base_block["DEVICE KERNEL DURATION [ns]"] = average_duration
            merged_blocks.append(base_block)
        else:
            # Non-collective ops: use MAX duration (critical path)
            max_duration_block = max(blocks, key=lambda x: x[1].get("DEVICE KERNEL DURATION [ns]", 0))
            merged_blocks.append(max_duration_block[1])

    return pd.DataFrame(merged_blocks)


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
    # Define columns to measure
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"

    # Step 1: Run device profiling (same as original)
    post_processed_results = run_device_perf(
        command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
    )

    # Step 2: Load and merge device rows (NEW STEP)
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    # Validate signpost filtering
    total_rows = len(df)
    signpost_rows = len(df[df["OP TYPE"] == "tt_signpost"])
    device_rows_before_filter = len(df[df["OP TYPE"] == "tt_dnn_device"])

    logger.info(f"CSV total rows: {total_rows}")
    logger.info(f"Signpost rows (will be excluded): {signpost_rows}")
    logger.info(f"Device operation rows: {device_rows_before_filter}")

    # Filter to device operations only (excludes signposts)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]

    # Validate signposts are excluded
    assert len(df) == device_rows_before_filter, "Filtering failed: unexpected row count"
    assert len(df[df["OP TYPE"] == "tt_signpost"]) == 0, "Signposts not properly filtered out"

    # Merge multi-device rows
    logger.info(f"Device rows before merge: {len(df)}")
    df_merged = merge_device_rows(df)
    logger.info(f"Device rows after merge: {len(df_merged)}")

    # Step 3: Recalculate metrics from merged data
    if not df_merged.empty:
        merged_kernel_durations = df_merged["DEVICE KERNEL DURATION [ns]"].dropna().tolist()
        if merged_kernel_durations:
            # Sum all operation durations for total execution time
            merged_sum_ns = sum(merged_kernel_durations)
            logger.info(f"Merged operations count: {len(merged_kernel_durations)}")
            logger.info(f"Merged sum (ns): {merged_sum_ns} ({merged_sum_ns / 1000:.1f} μs)")
            logger.info(f"Original post_processed_results[{inference_time_key}]: {post_processed_results.get(inference_time_key, 'N/A')}")
            post_processed_results[inference_time_key] = merged_sum_ns

    # Step 4: Validate against expected performance
    expected_perf_cols = {inference_time_key: expected_device_perf_ns_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=margin, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )

    # Step 5: Generate performance report
    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=comments,
    )


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ttnn_moe -k 'mesh-8x4-1600_no_pcc'",
            72600007,  # ~72.6ms - merged value for mesh-8x4
            "deepseek_v3_moe",
            "deepseek_v3_moe_mesh_8x4_2",
            1,
            1,
            0.03,  # 3% margin
            "seq_len_1600",
        ),
        (
            "pytest models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ttnn_moe -k 'mesh-8x4-3200_no_pcc'",
            101951824,  # ~102.0ms - merged value for mesh-8x4
            "deepseek_v3_moe",
            "deepseek_v3_moe_mesh_8x4_2",
            1,
            1,
            0.03,  # 3% margin
            "seq_len_3200",
        ),
    ],
    ids=[
        "mesh-8x4-2-seq1600_no_pcc",
        "mesh-8x4-2-seq3200_no_pcc",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_deepseek_v3_moe_perf_linear_8_2(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
):
    """
    Performance test for DeepSeek V3 MoE on mesh-8x4-2 topology.

    This test runs the full MoE pipeline (dispatch, routed experts, shared expert,
    combine, reduce) and measures device kernel execution time using Tracy profiler.

    Topology: 8 devices in linear configuration with 2 links per device
    Measured metric: Average device kernel duration in nanoseconds

    Args:
        command: Pytest command to execute the PCC test
        expected_device_perf_ns_per_iteration: Target performance in ns (dummy value initially)
        subdir: Output directory for performance reports
        model_name: Model identifier for tracking
        num_iterations: Number of profiling iterations
        batch_size: Batch size (currently 1)
        margin: Acceptable performance variance (0.03 = 3%)
        comments: Test case identifier (sequence length)
    """
    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
