# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
This file implements microbenchmarks for Wan2.2.
We will run microbenchmarks for two configurations: single galaxy and quad galaxy.
On single galaxy (4x8 torus), Wan is parallelized SP=8, TP=4.
On quad galaxy (4x32 torus), Wan is parallelized SP=32, TP=4.

Ops are grouped into the following groups:

## spatial layernorm
Distributed layernorm with dynamic affine parameters.

## spatial RMSNorm
Distributed RMSNorm on spatial QK. One variant is with fused rope, the other is without.

## prompt RMSNorm
Distributed RMSNorm on prompt K.

## spatial QKV
AllGather of spatial on TP axis followed by QKV projection.

## Ring Attention
RingAttention on spatial, which is SP and TP.

## Spatial dense out / unfused Q proj
AllGather of spatial on TP axis followed by dense output / unfused Q projection, both are same shape.

## prompt KV projection
Fused KV projection linear layer on prompt.

## Cross Attention
Cross attention of spatial Q against prompt KV.

## Spatial FF1
AllGather of spatial on TP axis followed by FF1.

## Spatial FF2
FF2 followed by reduce scatter on TP axis.
"""

import subprocess
import shlex

import pytest
import torch
import ttnn
import pandas as pd
from loguru import logger

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    get_profiler_folder,
)

from .....utils.tensor import bf16_tensor, bf16_tensor_2dshard
from .....layers.normalization import DistributedLayerNorm
from .....parallel.manager import CCLManager
from .....utils.test import line_params


# Wan2.2 14B model configuration
WAN_DIM = 5120
WAN_FFN_DIM = 13824
WAN_NUM_HEADS = 40
WAN_HEAD_DIM = WAN_DIM // WAN_NUM_HEADS
WAN_EPS = 1e-6

# 14B-720p generation sequence lengths
# Base sequence length: 75600 tokens (T=21, H=90, W=160, patch_size=(1,2,2))
# patch_F=21, patch_H=45, patch_W=80 => N = 21 * 45 * 80 = 75600

# Padded sequence lengths for different configurations
# Must be divisible by 32 * SP_size
SEQ_LEN_1XGLX = 75776  # 4x8 mesh, SP=8, divisible by 32*8=256
SEQ_LEN_4XGLX_EMULATED = 18944  # 75776 / 4, emulating quad galaxy workload on single galaxy

# Mesh configuration for BH Galaxy
SP_AXIS = 1  # Sequence parallel on axis 1
TP_AXIS = 0  # Tensor parallel on axis 0
NUM_LINKS = 2  # BH uses 2 links
TOPOLOGY = ttnn.Topology.Linear
MESH_SHAPE = (4, 8)

# Number of warmup and measurement iterations
NUM_WARMUP_ITERS = 3
NUM_MEASUREMENT_ITERS = 10

# CCL op names for determining aggregation method
CCL_OP_NAMES = [
    "AllGather",
    "ReduceScatter",
]

# Profiler output subdirectory
PROFILER_OUTPUT_DIR = "wan_microbench"

# Test module path for building pytest commands
TEST_MODULE = "models/experimental/tt_dit/tests/models/wan2_2/exabox/test_microbenchmark.py"

# Config name to test ID mapping
CONFIG_TO_TEST_ID = {
    "1xGLX_14b_720p": "1xGLX",
    "4xGLX_14b_720p_emulated": "4xGLX_emulated",
}


def run_device_profiler_quiet(
    command: str,
    output_logs_subdir: str,
    device_analysis_types: list[str] | None = None,
) -> None:
    """
    Run Tracy device profiler with output suppressed unless there's an error.

    Args:
        command: The pytest command to run
        output_logs_subdir: Subdirectory for profiler output
        device_analysis_types: List of analysis types to run
    """
    output_profiler_dir = get_profiler_folder(output_logs_subdir)

    device_analysis_opt = ""
    if device_analysis_types:
        device_analysis_opt = "".join([f" -a {analysis}" for analysis in device_analysis_types])

    profiler_cmd = (
        f"python3 -m tracy -p -r -o {output_profiler_dir} --check-exit-code "
        f"{device_analysis_opt} -t 5000 -m {shlex.quote(command)}"
    )

    result = subprocess.run(
        profiler_cmd,
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # On error, print the captured output
        print("=" * 80)
        print("PROFILER COMMAND FAILED")
        print("=" * 80)
        print(f"Command: {profiler_cmd}")
        print("-" * 80)
        print("STDOUT:")
        print(result.stdout)
        print("-" * 80)
        print("STDERR:")
        print(result.stderr)
        print("=" * 80)
        raise RuntimeError(f"Tracy profiler failed with return code {result.returncode}")


def is_ccl_op(op_name: str) -> bool:
    """Check if an operation is a CCL operation."""
    return any(ccl_name in op_name for ccl_name in CCL_OP_NAMES)


def aggregate_device_time(df: pd.DataFrame, op_name: str) -> float:
    """
    Aggregate device time for an operation across all devices.

    For non-CCL ops: use MAX device time across chips (worst case)
    For CCL ops: use MIN device time across chips (slowest device determines completion)

    Args:
        df: DataFrame with profiler results for a specific op
        op_name: Operation name for determining aggregation method

    Returns:
        Aggregated device time in microseconds
    """
    if df.empty:
        return 0.0

    durations_ns = df["DEVICE KERNEL DURATION [ns]"].astype(float)
    durations_us = durations_ns / 1000.0

    if is_ccl_op(op_name):
        return durations_us.min()
    else:
        return durations_us.max()


def analyze_op_group(
    output_logs_subdir: str,
    ops: list[tuple[str, str]],
    num_measurement_iters: int = NUM_MEASUREMENT_ITERS,
) -> dict:
    """
    Analyze a group of operations from Tracy profiler output.

    Args:
        output_logs_subdir: Path to profiler output directory
        ops: List of (op_code, short_name) tuples defining the op group
        num_measurement_iters: Number of measurement iterations that were run

    Returns:
        Dictionary with per-op times (keyed by short_name) and total time in microseconds
    """
    filename = get_latest_ops_log_filename(output_logs_subdir)
    df = pd.read_csv(filename)

    results = {}
    total_time_us = 0.0

    for op_name, short_name in ops:
        op_df = df[df["OP CODE"] == op_name]
        if op_df.empty:
            logger.warning(f"  {op_name}: NOT FOUND")
            results[short_name] = 0.0
            continue

        # Get all call counts and take the last N (measurement iterations)
        # Skip warmup iterations
        call_counts = sorted(op_df["GLOBAL CALL COUNT"].unique())
        if len(call_counts) > num_measurement_iters:
            measurement_call_counts = call_counts[-num_measurement_iters:]
        else:
            measurement_call_counts = call_counts

        # Aggregate across iterations
        iter_times = []
        for call_count in measurement_call_counts:
            call_df = op_df[op_df["GLOBAL CALL COUNT"] == call_count]
            agg_time = aggregate_device_time(call_df, op_name)
            iter_times.append(agg_time)

        avg_time_us = sum(iter_times) / len(iter_times) if iter_times else 0.0
        results[short_name] = avg_time_us
        total_time_us += avg_time_us

    results["total"] = total_time_us
    return results


def print_perf_table(
    title: str,
    config_name: str,
    seq_len: int,
    ops: list[tuple[str, str]],
    results: dict,
) -> None:
    """
    Print a formatted performance results table.

    Args:
        title: Table title (e.g., "SPATIAL LAYERNORM")
        config_name: Configuration name
        seq_len: Sequence length used
        ops: List of (op_code, short_name) tuples
        results: Dictionary with timing results keyed by short_name
    """
    print("\n" + "=" * 80)
    print(f"{title} PERFORMANCE: {config_name}")
    print("=" * 80)
    print(f"  Configuration: {config_name}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Mesh shape: {MESH_SHAPE}")
    print(f"  SP={MESH_SHAPE[SP_AXIS]}, TP={MESH_SHAPE[TP_AXIS]}")
    print("-" * 80)
    print(f"  {'Operation':<45} | {'Time (us)':>12} | {'Agg Method':>10}")
    print("-" * 80)

    for op_name, short_name in ops:
        agg_method = "min" if is_ccl_op(op_name) else "max"
        print(f"  {op_name:<45} | {results[short_name]:>12.2f} | {agg_method:>10}")

    print("-" * 80)
    print(f"  {'TOTAL':<45} | {results['total']:>12.2f} |")
    print("=" * 80 + "\n")


def build_test_command(test_name: str, config_name: str) -> str:
    """
    Build a pytest command for running a test with Tracy profiler.

    Args:
        test_name: Name of the test function (e.g., "test_run_spatial_layernorm")
        config_name: Configuration name

    Returns:
        Formatted pytest command string
    """
    test_id = CONFIG_TO_TEST_ID[config_name]
    return f"pytest {TEST_MODULE}::{test_name}[blackhole-{test_id}-bh_4x8] -v"


# =============================================================================
# Tests that run the actual operations (to be profiled with Tracy)
# =============================================================================


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [MESH_SHAPE, line_params],
    ],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "config_name, seq_len",
    [
        ("1xGLX_14b_720p", SEQ_LEN_1XGLX),
        ("4xGLX_14b_720p_emulated", SEQ_LEN_4XGLX_EMULATED),
    ],
    ids=["1xGLX", "4xGLX_emulated"],
)
def test_run_spatial_layernorm(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run spatial layernorm (norm1) in Wan transformer layer.

    This test is meant to be run with Tracy profiler via test_spatial_layernorm_perf.
    It runs the DistributedLayerNorm with dynamic affine parameters.
    """
    torch_dtype = torch.float32

    sp_factor = MESH_SHAPE[SP_AXIS]  # 8
    tp_factor = MESH_SHAPE[TP_AXIS]  # 4

    logger.info(f"Running spatial layernorm: {config_name}")
    logger.info(f"  Sequence length: {seq_len}")
    logger.info(f"  SP factor: {sp_factor}, TP factor: {tp_factor}")

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=NUM_LINKS,
        topology=TOPOLOGY,
    )

    # Create DistributedLayerNorm (norm1 in transformer block)
    norm = DistributedLayerNorm(
        embedding_dim=WAN_DIM,
        norm_eps=WAN_EPS,
        norm_elementwise_affine=False,
        bias=False,
        mesh_axis=TP_AXIS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
    )

    # Create input tensors
    torch.manual_seed(0)
    batch_size = 1

    spatial_torch = torch.randn((1, batch_size, seq_len, WAN_DIM), dtype=torch_dtype)
    dynamic_weight_torch = torch.randn((1, batch_size, 1, WAN_DIM), dtype=torch_dtype)
    dynamic_bias_torch = torch.randn((1, batch_size, 1, WAN_DIM), dtype=torch_dtype)

    # Convert to TT tensors with proper sharding
    tt_spatial = bf16_tensor_2dshard(
        spatial_torch,
        device=mesh_device,
        shard_mapping={SP_AXIS: 2, TP_AXIS: 3},
    )
    tt_dynamic_weight = bf16_tensor(
        dynamic_weight_torch,
        device=mesh_device,
        mesh_axis=TP_AXIS,
        shard_dim=3,
    )
    tt_dynamic_bias = bf16_tensor(
        dynamic_bias_torch,
        device=mesh_device,
        mesh_axis=TP_AXIS,
        shard_dim=3,
    )

    logger.info(f"  TT spatial shape: {tt_spatial.shape}")

    # Warmup iterations
    for _ in range(NUM_WARMUP_ITERS):
        _ = norm(tt_spatial, dynamic_weight=tt_dynamic_weight, dynamic_bias=tt_dynamic_bias)
    ttnn.synchronize_device(mesh_device)

    # Measurement iterations
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = norm(tt_spatial, dynamic_weight=tt_dynamic_weight, dynamic_bias=tt_dynamic_bias)
    ttnn.synchronize_device(mesh_device)

    logger.info(f"Completed {NUM_WARMUP_ITERS} warmup + {NUM_MEASUREMENT_ITERS} measurement iterations")


# =============================================================================
# Op group definitions
# =============================================================================

# Spatial layernorm: DistributedLayerNorm with dynamic affine parameters
SPATIAL_LAYERNORM_OPS = [
    ("PreAllGatherDeviceOperation", "pre_allgather"),
    ("AllGatherAsyncDeviceOperation", "all_gather"),
    ("PostAllGatherDeviceOperation", "post_allgather"),
]


# =============================================================================
# Performance tests (run with Tracy profiler)
# =============================================================================


@pytest.mark.parametrize(
    "config_name, seq_len",
    [
        ("1xGLX_14b_720p", SEQ_LEN_1XGLX),
        ("4xGLX_14b_720p_emulated", SEQ_LEN_4XGLX_EMULATED),
    ],
    ids=["1xGLX", "4xGLX_emulated"],
)
def test_spatial_layernorm_perf(config_name: str, seq_len: int) -> None:
    """
    Measure device performance for spatial layernorm using Tracy profiler.

    This test:
    1. Runs test_run_spatial_layernorm with Tracy profiler
    2. Parses the profiler output
    3. Aggregates device times (max for non-CCL, min for CCL ops)
    4. Prints a performance summary table
    """
    command = build_test_command("test_run_spatial_layernorm", config_name)

    run_device_profiler_quiet(
        command,
        PROFILER_OUTPUT_DIR,
        device_analysis_types=["device_kernel_duration"],
    )

    results = analyze_op_group(PROFILER_OUTPUT_DIR, SPATIAL_LAYERNORM_OPS)
    print_perf_table("SPATIAL LAYERNORM", config_name, seq_len, SPATIAL_LAYERNORM_OPS, results)
