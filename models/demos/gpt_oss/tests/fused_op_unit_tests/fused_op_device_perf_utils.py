# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for fused op device performance tests.

This module provides common functionality for collecting device performance metrics
using Tracy profiler and aggregating results into BenchmarkData format.
"""

import math
import os
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

# ==============================================================================
# Constants
# ==============================================================================
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100


@dataclass
class DevicePerfResult:
    """Results from a device performance measurement."""

    op_name: str
    step_name: str
    total_kernel_us: float
    total_op_to_op_us: float
    avg_kernel_us: float
    avg_op_to_op_us: float
    op_stats: dict[str, dict[str, float]]
    batch_size: int
    seq_len: int


def merge_device_rows_for_perf(df: pd.DataFrame) -> pd.DataFrame:
    """Merge device rows for performance analysis.

    For CCL ops (AllGather, ReduceScatter, AllReduce, AllToAll), use average duration.
    For other ops, use max duration across devices.
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
                f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices {missing_devices} - do not trust data for this op or directly subsequent ops with the same name"
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
    command: str, subdir: str, warmup_iters: int = 0, use_signposts: bool = False
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Collect device performance metrics using Tracy profiler.

    Args:
        command: pytest command to run with profiling
        subdir: subdirectory for profiler output
        warmup_iters: number of warmup iterations to skip in analysis
        use_signposts: whether to use signpost markers to filter data

    Returns:
        Tuple of (op_stats dict, total_kernel_ns, total_op_to_op_ns)
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
            "total_kernel_duration_ns": sum(kernel_vals),
            "total_op_to_op_latency_ns": sum(op_to_op_vals),
            "call_count": len(kernel_vals),
        }

    total_kernel_ns = sum(entry["total_kernel_duration_ns"] for entry in op_stats.values())
    total_op_to_op_ns = sum(entry["total_op_to_op_latency_ns"] for entry in op_stats.values())
    return op_stats, total_kernel_ns, total_op_to_op_ns


def run_single_op_device_perf(
    op_name: str,
    test_path: str,
    test_function: str,
    env_var: str,
    mode: str,
    seq_len: int,
    batch_size: int,
    subdir: str = "gpt_oss_fused_ops_device_perf",
    use_trace: bool = True,
) -> DevicePerfResult:
    """Run device performance test for a single fused op.

    Args:
        op_name: Human-readable name of the op (e.g., "experts", "router")
        test_path: Path to the test file
        test_function: Name of the test function to run
        env_var: Environment variable to set for device perf mode
        mode: "decode" or "prefill"
        seq_len: Sequence length
        batch_size: Batch size
        subdir: Subdirectory for profiler output
        use_trace: Whether to use trace mode filter

    Returns:
        DevicePerfResult with all metrics
    """
    step_name = f"gpt_oss_{op_name}_device_perf_{mode}_seq{seq_len}"
    trace_filter = "trace" if use_trace and mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::{test_function} -k "{expr}"'

    logger.info(f"Running device perf for {op_name}: {command}")

    os.environ[env_var] = "1"
    try:
        op_stats, total_kernel_ns, total_op_to_op_ns = collect_device_perf(
            command,
            subdir=subdir,
            warmup_iters=0,
            use_signposts=True,
        )
    finally:
        os.environ.pop(env_var, None)

    assert op_stats, f"No device perf stats captured for {op_name}."

    total_kernel_us = total_kernel_ns / 1000.0
    total_op_to_op_us = total_op_to_op_ns / 1000.0
    avg_kernel_us = total_kernel_us / DEVICE_PERF_ITERS
    avg_op_to_op_us = total_op_to_op_us / DEVICE_PERF_ITERS

    logger.info(
        f"{op_name} device perf totals ({DEVICE_PERF_ITERS} iterations): "
        f"kernel={total_kernel_us:.3f} us, op_to_op={total_op_to_op_us:.3f} us"
    )
    logger.info(
        f"{op_name} device perf per-iteration averages: "
        f"kernel={avg_kernel_us:.3f} us, op_to_op={avg_op_to_op_us:.3f} us"
    )

    return DevicePerfResult(
        op_name=op_name,
        step_name=step_name,
        total_kernel_us=total_kernel_us,
        total_op_to_op_us=total_op_to_op_us,
        avg_kernel_us=avg_kernel_us,
        avg_op_to_op_us=avg_op_to_op_us,
        op_stats=op_stats,
        batch_size=batch_size,
        seq_len=seq_len,
    )


def add_result_to_benchmark(
    benchmark_data: BenchmarkData,
    perf_profiler: BenchmarkProfiler,
    result: DevicePerfResult,
) -> None:
    """Add a DevicePerfResult to BenchmarkData.

    Args:
        benchmark_data: BenchmarkData instance to add measurements to
        perf_profiler: BenchmarkProfiler instance
        result: DevicePerfResult with metrics to add
    """
    benchmark_data.add_measurement(
        perf_profiler,
        0,
        result.step_name,
        "total_kernel_duration_us",
        result.total_kernel_us,
    )
    benchmark_data.add_measurement(
        perf_profiler,
        0,
        result.step_name,
        "total_op_to_op_latency_us",
        result.total_op_to_op_us,
    )
    benchmark_data.add_measurement(
        perf_profiler,
        0,
        result.step_name,
        "avg_kernel_duration_us",
        result.avg_kernel_us,
    )
    benchmark_data.add_measurement(
        perf_profiler,
        0,
        result.step_name,
        "avg_op_to_op_latency_us",
        result.avg_op_to_op_us,
    )


# ==============================================================================
# Fused Op Registry
# ==============================================================================
# Registry of all fused ops with their test configurations
FUSED_OP_CONFIGS = {
    "experts": {
        "test_path": "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_experts.py",
        "test_function": "test_gpt_oss_experts",
        "env_var": "GPT_OSS_EXPERTS_DEVICE_PERF",
        "batch_size": 128,
        "use_trace": True,
        "description": "Full decode_forward including CCLs (all_to_all_dispatch, combine, all_reduce)",
    },
    "prepare_expert_tensors": {
        "test_path": "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_prepare_expert_tensors.py",
        "test_function": "test_gpt_oss_prepare_expert_tensors",
        "env_var": "GPT_OSS_PREPARE_EXPERT_TENSORS_DEVICE_PERF",
        "batch_size": 32,  # Per-device batch size
        "use_trace": True,
        "description": "Tensor preparation (reshape, typecast, layout conversion) before dispatch",
    },
    "prepare_expert_weights": {
        "test_path": "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_prepare_expert_weights.py",
        "test_function": "test_gpt_oss_prepare_expert_weights",
        "env_var": "GPT_OSS_PREPARE_EXPERT_WEIGHTS_DEVICE_PERF",
        "batch_size": 128,
        "use_trace": True,
        "description": "Routing weight preparation (reshape, repeat, permute)",
    },
    "router": {
        "test_path": "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_router.py",
        "test_function": "test_gpt_oss_router",
        "env_var": "GPT_OSS_ROUTER_DEVICE_PERF",
        "batch_size": 128,
        "use_trace": True,
        "description": "TopK router (linear + topk + softmax)",
    },
    "experts_mlp": {
        "test_path": "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_experts_mlp.py",
        "test_function": "test_gpt_oss_experts_mlp",
        "env_var": "GPT_OSS_EXPERTS_MLP_DEVICE_PERF",
        "batch_size": 128,
        "use_trace": False,  # MLP uses eager mode for device perf measurement
        "description": "Expert MLP computation (batched matmul + SwiGLU)",
    },
}


def get_all_fused_op_names() -> list[str]:
    """Get list of all registered fused op names."""
    return list(FUSED_OP_CONFIGS.keys())


def get_fused_op_config(op_name: str) -> dict:
    """Get configuration for a specific fused op."""
    if op_name not in FUSED_OP_CONFIGS:
        raise ValueError(f"Unknown fused op: {op_name}. Available: {list(FUSED_OP_CONFIGS.keys())}")
    return FUSED_OP_CONFIGS[op_name]
