# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utility functions for DeepSeek V3 fused op unit tests.

This module contains common helper functions used across all fused op unit tests
to reduce code duplication and ensure consistent behavior.
"""

import math
import os
from collections import defaultdict
from typing import Callable

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler


def get_int_env(name: str, default: int) -> int:
    """Get an integer environment variable with a default value.

    Args:
        name: Name of the environment variable.
        default: Default value if the environment variable is not set.

    Returns:
        The integer value of the environment variable, or the default.

    Raises:
        ValueError: If the environment variable is set but cannot be parsed as an integer.
    """
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as e:
        raise ValueError(f"Env var {name} must be an int, got {val!r}") from e


def maybe_skip_long_seq(seq_len: int, long_seq_env_var: str, threshold: int = 8192):
    """Skip test if sequence length exceeds threshold and env var is not set.

    Args:
        seq_len: The sequence length being tested.
        long_seq_env_var: Name of the environment variable that enables long sequence tests.
        threshold: Maximum sequence length before requiring the env var (default: 8192).
    """
    if seq_len <= threshold:
        return
    if os.getenv(long_seq_env_var) is None:
        pytest.skip(f"Set {long_seq_env_var}=1 to enable seq_len={seq_len} coverage.")


def compare_with_reference(
    tt_output: torch.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    atol: float,
    rtol: float,
    convert_to_float: bool = False,
    strict_assert: bool = True,
) -> tuple[float, float]:
    """Compare TTNN output with reference and return metrics.

    Args:
        tt_output: Output tensor from TTNN operation.
        ref_output: Reference output tensor.
        expected_pcc: Minimum expected Pearson correlation coefficient.
        atol: Absolute tolerance for torch.testing.assert_close.
        rtol: Relative tolerance for torch.testing.assert_close.
        convert_to_float: If True, convert tensors to float before comparison.
        strict_assert: If True, raise AssertionError on assert_close failure.
                      If False, log a warning instead.

    Returns:
        Tuple of (pcc_value, max_abs_error) for logging to superset.
    """
    if convert_to_float:
        tt_for_compare = tt_output.float()
        ref_for_compare = ref_output.float()
    else:
        tt_for_compare = tt_output
        ref_for_compare = ref_output

    passing, pcc = comp_pcc(ref_for_compare, tt_for_compare, expected_pcc)
    max_abs_error = (tt_for_compare - ref_for_compare).abs().max().item()
    logger.info(f"PCC: {pcc}")
    logger.info(f"Max absolute error: {max_abs_error}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"

    if strict_assert:
        torch.testing.assert_close(tt_for_compare, ref_for_compare, rtol=rtol, atol=atol)
    else:
        try:
            torch.testing.assert_close(tt_for_compare, ref_for_compare, rtol=rtol, atol=atol)
        except AssertionError as e:
            logger.warning(f"assert_close failed but PCC passed: {e}")

    return pcc, max_abs_error


def log_run_mode(
    mode: str,
    trace_mode: bool,
    program_cache_enabled: bool,
    seq_len: int,
    **extra_fields,
):
    """Log the test run configuration.

    Args:
        mode: Test mode ("decode" or "prefill").
        trace_mode: Whether trace mode is enabled.
        program_cache_enabled: Whether program cache is enabled.
        seq_len: Sequence length being tested.
        **extra_fields: Additional fields to log (e.g., use_real_weights, hf_config_size_attr).
    """
    logger.info("=== TEST RUN CONFIGURATION ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Sequence length: {seq_len}")
    for key, value in extra_fields.items():
        # Convert snake_case to readable format
        readable_key = key.replace("_", " ").title()
        logger.info(f"{readable_key}: {value}")
    logger.info(f"Trace mode: {trace_mode}")
    logger.info(f"Program cache enabled: {program_cache_enabled}")
    logger.info("===============================")


def deallocate_outputs(outputs):
    """Deallocate outputs which can be a single tensor or tuple of tensors.

    Args:
        outputs: A single ttnn.Tensor or a tuple of ttnn.Tensor objects.
    """
    if isinstance(outputs, tuple):
        for out in outputs:
            ttnn.deallocate(out)
    else:
        ttnn.deallocate(outputs)


def measure_perf_us(
    mesh_device: ttnn.MeshDevice,
    op_fn: Callable,
    warmup_iters: int,
    measure_iters: int,
    trace_mode: bool = False,
    profiler_name: str = "perf_measurement",
) -> float:
    """Measure operation performance in microseconds.

    Args:
        mesh_device: The mesh device to run operations on.
        op_fn: Function that executes the operation. Should return output tensor(s).
               For trace mode with persistent buffers, should accept optional
               `persistent_output_buffer` kwarg.
        warmup_iters: Number of warmup iterations before measurement.
        measure_iters: Number of iterations to measure.
        trace_mode: If True, use trace capture and replay for measurement.
        profiler_name: Name for the profiler measurement (default: "perf_measurement").

    Returns:
        Average operation time in microseconds.
    """
    ttnn.synchronize_device(mesh_device)

    if trace_mode:
        # Warmup in eager mode first
        for _ in range(warmup_iters):
            outputs = op_fn()
            ttnn.synchronize_device(mesh_device)
            deallocate_outputs(outputs)

        # Capture trace
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_outputs = op_fn()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        deallocate_outputs(traced_outputs)

        # Additional warmup with trace execution
        for _ in range(warmup_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)

        # Measure with trace
        profiler.clear()
        profiler.start(profiler_name)
        for _ in range(measure_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
        profiler.end(profiler_name, PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get(profiler_name) * 1e6

    # Eager mode
    for _ in range(warmup_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        deallocate_outputs(outputs)

    profiler.clear()
    profiler.start(profiler_name)
    for _ in range(measure_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        deallocate_outputs(outputs)
    profiler.end(profiler_name, PERF_CNT=measure_iters)
    return profiler.get(profiler_name) * 1e6


def merge_device_rows_for_perf(df: pd.DataFrame) -> pd.DataFrame:
    """Merge device performance rows across devices for collective operations.

    For collective operations (AllGather, ReduceScatter, etc.), computes average
    kernel duration across devices. For other operations, takes the maximum
    kernel duration.

    Args:
        df: DataFrame containing device performance data with columns:
            - "OP CODE": Operation name
            - "OP TYPE": Operation type (e.g., "tt_dnn_device")
            - "DEVICE ID": Device identifier
            - "DEVICE KERNEL DURATION [ns]": Kernel duration in nanoseconds

    Returns:
        DataFrame with merged performance rows.
    """
    block_by_device = defaultdict(list)

    for _, row in df.iterrows():
        op_name = row["OP CODE"]
        op_type = row["OP TYPE"]

        if op_type == "tt_dnn_device":
            device_id = int(row["DEVICE ID"])
            block_by_device[device_id].append((op_name, row.to_dict()))

    device_ids = sorted(block_by_device.keys())
    merged_blocks = []
    global_index = 0
    while max(len(block_by_device[device_id]) for device_id in device_ids) > 0:
        blocks = []
        op_name = None
        missing_devices = []
        for device_id in device_ids:
            if not len(block_by_device[device_id]):
                logger.warning(f"Warning: Device {device_id} is missing operation {op_name} at index {global_index}")
                continue
            if op_name is None:
                op_name = block_by_device[device_id][0][0]
            elif op_name != block_by_device[device_id][0][0]:
                missing_devices.append(device_id)
                continue

            blocks.append(block_by_device[device_id].pop(0))

        if missing_devices:
            logger.warning(
                f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices "
                f"{missing_devices} - do not trust data for this op or directly subsequent ops with the same name"
            )

        if not blocks:
            break

        is_collective = any(tag in op_name for tag in ("AllGather", "ReduceScatter", "AllReduce", "AllToAll"))
        if is_collective:
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
            max_duration_block = max(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(max_duration_block[1])

        global_index += 1

    return pd.DataFrame(merged_blocks)


def collect_device_perf(
    command: str,
    subdir: str,
    warmup_iters: int,
    use_signposts: bool = False,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Collect device performance statistics by running a profiled command.

    Args:
        command: The pytest command to run with device profiling.
        subdir: Subdirectory for profiler output files.
        warmup_iters: Number of warmup iterations to exclude from statistics.
        use_signposts: If True, only analyze ops between "start" and "stop" signposts.

    Returns:
        Tuple of:
            - op_stats: Dict mapping op names to their avg kernel duration and op-to-op latency.
            - total_kernel_ns: Total kernel duration in nanoseconds.
            - total_op_to_op_ns: Total op-to-op latency in nanoseconds.
    """
    device_analysis_types = ["device_kernel_duration"]
    run_device_profiler(
        command,
        subdir,
        device_analysis_types=device_analysis_types,
        op_support_count=10000,
    )
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    if use_signposts:
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        assert not markers.empty, "No signposts found in device perf log."
        start_indices = markers[markers == "start"].index
        stop_indices = markers[markers == "stop"].index
        assert not start_indices.empty, "Missing signpost 'start' in device perf log."
        assert not stop_indices.empty, "Missing signpost 'stop' in device perf log."
        start_idx = start_indices[0]
        stop_idx = stop_indices[-1]
        assert start_idx < stop_idx, "Signpost 'stop' must come after 'start'."
        df = df.iloc[start_idx + 1 : stop_idx]

    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows_for_perf(df)

    required_cols = ["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    assert not missing_cols, f"Missing device perf columns: {missing_cols}"

    df["DEVICE KERNEL DURATION [ns]"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce").fillna(0.0)
    df["OP TO OP LATENCY [ns]"] = pd.to_numeric(df["OP TO OP LATENCY [ns]"], errors="coerce").fillna(0.0)

    op_stats: dict[str, dict[str, float]] = {}
    for op_code, group in df.groupby("OP CODE"):
        kernel_vals = group["DEVICE KERNEL DURATION [ns]"].tolist()
        op_to_op_vals = group["OP TO OP LATENCY [ns]"].tolist()
        if warmup_iters > 0:
            kernel_vals = kernel_vals[warmup_iters:]
            op_to_op_vals = op_to_op_vals[warmup_iters:]
        assert kernel_vals, f"No kernel duration samples for op {op_code}"
        assert op_to_op_vals, f"No op-to-op latency samples for op {op_code}"
        op_stats[op_code] = {
            "avg_kernel_duration_ns": sum(kernel_vals) / len(kernel_vals),
            "avg_op_to_op_latency_ns": sum(op_to_op_vals) / len(op_to_op_vals),
        }

    total_kernel_ns = sum(entry["avg_kernel_duration_ns"] for entry in op_stats.values())
    total_op_to_op_ns = sum(entry["avg_op_to_op_latency_ns"] for entry in op_stats.values())
    return op_stats, total_kernel_ns, total_op_to_op_ns


def skip_single_device_ccl(op_name: str):
    """Skip test because it includes CCL operations that require multiple devices.

    Args:
        op_name: Name of the operation being tested.
    """
    pytest.skip(f"Single-device test is not applicable because {op_name} includes CCL ops.")


def skip_single_device_sharded(op_name: str):
    """Skip test because it relies on width-sharded matmuls across the mesh.

    Args:
        op_name: Name of the operation being tested.
    """
    pytest.skip(
        f"Single-device test is not applicable because {op_name} relies on width-sharded matmuls across the mesh."
    )
