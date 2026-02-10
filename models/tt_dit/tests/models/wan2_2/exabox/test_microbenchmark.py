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

import shlex
import subprocess

import pandas as pd
import pytest
import torch
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename, get_profiler_folder

import ttnn

from .....layers.linear import ColParallelLinear, RowParallelLinear
from .....layers.normalization import DistributedLayerNorm, DistributedRMSNorm
from .....parallel.manager import CCLManager
from .....utils.tensor import bf16_tensor, bf16_tensor_2dshard
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

# Prompt sequence length (text tokens, replicated across SP)
PROMPT_SEQ_LEN = 512

# Mesh configuration for BH Galaxy
SP_AXIS = 1  # Sequence parallel on axis 1
TP_AXIS = 0  # Tensor parallel on axis 0
NUM_LINKS = 2  # BH uses 2 links
TOPOLOGY = ttnn.Topology.Linear
MESH_SHAPE = (4, 8)

# Number of warmup and measurement iterations
NUM_WARMUP_ITERS = 3
NUM_MEASUREMENT_ITERS = 10

# CCL op names for determining aggregation method (use MIN across devices)
CCL_OP_NAMES = [
    "AllGather",
    "ReduceScatter",
    "RingJoint",  # Ring attention involves CCL
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

# Common pytest parametrize decorators
MESH_DEVICE_PARAMS = pytest.mark.parametrize(
    "mesh_device, device_params",
    [[MESH_SHAPE, line_params]],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)

CONFIG_WITH_SEQ_LEN_PARAMS = pytest.mark.parametrize(
    "config_name, seq_len",
    [
        ("1xGLX_14b_720p", SEQ_LEN_1XGLX),
        ("4xGLX_14b_720p_emulated", SEQ_LEN_4XGLX_EMULATED),
    ],
    ids=["1xGLX", "4xGLX_emulated"],
)

CONFIG_ONLY_PARAMS = pytest.mark.parametrize(
    "config_name",
    ["1xGLX_14b_720p", "4xGLX_14b_720p_emulated"],
    ids=["1xGLX", "4xGLX_emulated"],
)


# =============================================================================
# Helper functions
# =============================================================================


def create_ccl_manager(mesh_device: ttnn.MeshDevice) -> CCLManager:
    """Create a CCL manager with standard configuration."""
    return CCLManager(
        mesh_device=mesh_device,
        num_links=NUM_LINKS,
        topology=TOPOLOGY,
    )


def create_distributed_rmsnorm(mesh_device: ttnn.MeshDevice, ccl_manager: CCLManager) -> DistributedRMSNorm:
    """Create a DistributedRMSNorm with standard configuration and dummy weights."""
    norm = DistributedRMSNorm(
        embedding_dim=WAN_DIM,
        norm_eps=WAN_EPS,
        norm_elementwise_affine=True,
        bias=False,
        mesh_axis=TP_AXIS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
    )
    # Load dummy weights
    torch.manual_seed(0)
    norm.load_state_dict({"weight": torch.randn(WAN_DIM)})
    return norm


def create_col_parallel_linear(
    mesh_device: ttnn.MeshDevice,
    ccl_manager: CCLManager,
    in_features: int = WAN_DIM,
    out_features: int = WAN_DIM,
) -> ColParallelLinear:
    """Create a ColParallelLinear with standard configuration and dummy weights."""
    linear = ColParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        mesh_device=mesh_device,
        mesh_axis=TP_AXIS,
        ccl_manager=ccl_manager,
    )
    # Load dummy weights
    torch.manual_seed(0)
    tp_factor = MESH_SHAPE[TP_AXIS]
    linear.load_state_dict(
        {
            "weight": torch.randn(out_features, in_features),
            "bias": torch.randn(out_features),
        }
    )
    return linear


def run_perf_test(test_name: str, config_name: str, seq_len: int, title: str, ops: list) -> None:
    """
    Run a performance test with Tracy profiler and print results.

    Args:
        test_name: Name of the test_run_* function to profile
        config_name: Configuration name (1xGLX or 4xGLX)
        seq_len: Sequence length for display
        title: Title for the results table
        ops: List of (op_code, short_name) tuples
    """
    command = build_test_command(test_name, config_name)

    run_device_profiler_quiet(
        command,
        PROFILER_OUTPUT_DIR,
        device_analysis_types=["device_kernel_duration"],
    )

    results = analyze_op_group(PROFILER_OUTPUT_DIR, ops)
    print_perf_table(title, config_name, seq_len, ops, results)


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
    ops: list[tuple[str, str] | tuple[str, str, int]],
    num_measurement_iters: int = NUM_MEASUREMENT_ITERS,
) -> dict:
    """
    Analyze a group of operations from Tracy profiler output.

    Args:
        output_logs_subdir: Path to profiler output directory
        ops: List of (op_code, short_name) or (op_code, short_name, calls_per_iter) tuples.
             calls_per_iter defaults to 1 if not specified.
        num_measurement_iters: Number of measurement iterations that were run

    Returns:
        Dictionary with per-op times (keyed by short_name) and total time in microseconds
    """
    filename = get_latest_ops_log_filename(output_logs_subdir)
    df = pd.read_csv(filename)

    results = {}
    total_time_us = 0.0

    for op_entry in ops:
        if len(op_entry) == 3:
            op_name, short_name, calls_per_iter = op_entry
        else:
            op_name, short_name = op_entry
            calls_per_iter = 1

        op_df = df[df["OP CODE"] == op_name]
        if op_df.empty:
            logger.warning(f"  {op_name}: NOT FOUND")
            results[short_name] = 0.0
            continue

        # Get all call counts and take the last N * calls_per_iter (measurement iterations)
        # Skip warmup iterations
        call_counts = sorted(op_df["GLOBAL CALL COUNT"].unique())
        num_calls_to_analyze = num_measurement_iters * calls_per_iter
        if len(call_counts) > num_calls_to_analyze:
            measurement_call_counts = call_counts[-num_calls_to_analyze:]
        else:
            measurement_call_counts = call_counts

        # Aggregate across all calls (sum times for multiple calls per iter)
        iter_times = []
        for call_count in measurement_call_counts:
            call_df = op_df[op_df["GLOBAL CALL COUNT"] == call_count]
            agg_time = aggregate_device_time(call_df, op_name)
            iter_times.append(agg_time)

        # Sum all call times and divide by number of iterations
        total_call_time = sum(iter_times) if iter_times else 0.0
        avg_time_us = total_call_time / num_measurement_iters if iter_times else 0.0
        results[short_name] = avg_time_us
        total_time_us += avg_time_us

    results["total"] = total_time_us
    return results


def print_perf_table(
    title: str,
    config_name: str,
    seq_len: int,
    ops: list[tuple[str, str] | tuple[str, str, int]],
    results: dict,
) -> None:
    """
    Print a formatted performance results table.

    Args:
        title: Table title (e.g., "SPATIAL LAYERNORM")
        config_name: Configuration name
        seq_len: Sequence length used
        ops: List of (op_code, short_name) or (op_code, short_name, calls_per_iter) tuples
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

    for op_entry in ops:
        op_name = op_entry[0]
        short_name = op_entry[1]
        calls_per_iter = op_entry[2] if len(op_entry) == 3 else 1
        agg_method = "min" if is_ccl_op(op_name) else "max"
        display_name = f"{op_name} (x{calls_per_iter})" if calls_per_iter > 1 else op_name
        print(f"  {display_name:<45} | {results[short_name]:>12.2f} | {agg_method:>10}")

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


@MESH_DEVICE_PARAMS
@CONFIG_WITH_SEQ_LEN_PARAMS
def test_run_spatial_layernorm(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run spatial layernorm (norm1) in Wan transformer layer.
    Profiled via test_spatial_layernorm_perf.
    """
    logger.info(f"Running spatial layernorm: {config_name}, seq_len={seq_len}")

    ccl_manager = create_ccl_manager(mesh_device)

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

    spatial_torch = torch.randn((1, batch_size, seq_len, WAN_DIM), dtype=torch.float32)
    dynamic_weight_torch = torch.randn((1, batch_size, 1, WAN_DIM), dtype=torch.float32)
    dynamic_bias_torch = torch.randn((1, batch_size, 1, WAN_DIM), dtype=torch.float32)

    tt_spatial = bf16_tensor_2dshard(spatial_torch, device=mesh_device, shard_mapping={SP_AXIS: 2, TP_AXIS: 3})
    tt_dynamic_weight = bf16_tensor(dynamic_weight_torch, device=mesh_device, mesh_axis=TP_AXIS, shard_dim=3)
    tt_dynamic_bias = bf16_tensor(dynamic_bias_torch, device=mesh_device, mesh_axis=TP_AXIS, shard_dim=3)

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = norm(tt_spatial, dynamic_weight=tt_dynamic_weight, dynamic_bias=tt_dynamic_bias)
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = norm(tt_spatial, dynamic_weight=tt_dynamic_weight, dynamic_bias=tt_dynamic_bias)
    ttnn.synchronize_device(mesh_device)


@MESH_DEVICE_PARAMS
@CONFIG_WITH_SEQ_LEN_PARAMS
def test_run_spatial_rmsnorm(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run spatial RMSNorm (norm_q/norm_k) with fused RoPE in Wan self-attention.
    Profiled via test_spatial_rmsnorm_perf.
    """
    tp_factor = MESH_SHAPE[TP_AXIS]
    n_local_heads = WAN_NUM_HEADS // tp_factor
    local_dim = WAN_DIM // tp_factor
    seq_len_local = seq_len // MESH_SHAPE[SP_AXIS]

    logger.info(f"Running spatial RMSNorm: {config_name}, seq_len={seq_len}")

    ccl_manager = create_ccl_manager(mesh_device)
    norm = create_distributed_rmsnorm(mesh_device, ccl_manager)

    # Create input tensor [1, B, N_local, D_local]
    batch_size = 1
    input_torch = torch.randn((1, batch_size, seq_len_local, local_dim), dtype=torch.float32)
    tt_input = bf16_tensor(input_torch, device=mesh_device)

    # Create RoPE tensors
    rope_cos_torch = torch.randn((1, batch_size, seq_len_local, WAN_HEAD_DIM), dtype=torch.float32)
    rope_sin_torch = torch.randn((1, batch_size, seq_len_local, WAN_HEAD_DIM), dtype=torch.float32)
    tt_rope_cos = bf16_tensor(rope_cos_torch, device=mesh_device)
    tt_rope_sin = bf16_tensor(rope_sin_torch, device=mesh_device)

    # Transformation matrix [1, 1, 32, 32]
    trans_mat_torch = torch.eye(32, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tt_trans_mat = bf16_tensor(trans_mat_torch, device=mesh_device)

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = norm(
            tt_input,
            num_heads_per_device=n_local_heads,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
            trans_mat=tt_trans_mat,
        )
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = norm(
            tt_input,
            num_heads_per_device=n_local_heads,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
            trans_mat=tt_trans_mat,
        )
    ttnn.synchronize_device(mesh_device)


@MESH_DEVICE_PARAMS
@CONFIG_ONLY_PARAMS
def test_run_prompt_rmsnorm(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    reset_seeds,
) -> None:
    """
    Run prompt RMSNorm (norm_k on prompt K) without RoPE in Wan cross-attention.
    Profiled via test_prompt_rmsnorm_perf.
    """
    tp_factor = MESH_SHAPE[TP_AXIS]
    n_local_heads = WAN_NUM_HEADS // tp_factor
    local_dim = WAN_DIM // tp_factor

    logger.info(f"Running prompt RMSNorm: {config_name}, prompt_len={PROMPT_SEQ_LEN}")

    ccl_manager = create_ccl_manager(mesh_device)
    norm = create_distributed_rmsnorm(mesh_device, ccl_manager)

    # Create input tensor [1, B, L, D_local]
    batch_size = 1
    input_torch = torch.randn((1, batch_size, PROMPT_SEQ_LEN, local_dim), dtype=torch.float32)
    tt_input = bf16_tensor(input_torch, device=mesh_device)

    # Warmup (no RoPE for cross-attention)
    for _ in range(NUM_WARMUP_ITERS):
        _ = norm(tt_input, num_heads_per_device=n_local_heads)
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = norm(tt_input, num_heads_per_device=n_local_heads)
    ttnn.synchronize_device(mesh_device)


def get_ring_sdpa_configs(mesh_device: ttnn.MeshDevice, config_name: str):
    """Get program and compute configs for ring SDPA."""
    full_grid = mesh_device.compute_with_storage_grid_size()
    sdpa_worker_grid = (full_grid.x, full_grid.y - 1)

    # Chunk sizes: 1xGLX (SP=8) uses (128, 512), 4xGLX (SP=32) uses (128, 128)
    if "4xGLX" in config_name:
        q_chunk_size, k_chunk_size = 128, 128
    else:
        q_chunk_size, k_chunk_size = 128, 512

    ring_sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_worker_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )

    return ring_sdpa_program_config, sdpa_compute_kernel_config, sdpa_worker_grid


@MESH_DEVICE_PARAMS
@CONFIG_WITH_SEQ_LEN_PARAMS
def test_run_spatial_qkv(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run spatial QKV: AllGather on TP axis followed by Q, K, V projections.
    Profiled via test_spatial_qkv_perf.
    """
    tp_factor = MESH_SHAPE[TP_AXIS]
    sp_factor = MESH_SHAPE[SP_AXIS]
    seq_len_local = seq_len // sp_factor

    logger.info(f"Running spatial QKV: {config_name}, seq_len={seq_len}")

    ccl_manager = create_ccl_manager(mesh_device)

    # Create Q, K, V projection layers (ColParallelLinear)
    to_q = create_col_parallel_linear(mesh_device, ccl_manager)
    to_k = create_col_parallel_linear(mesh_device, ccl_manager)
    to_v = create_col_parallel_linear(mesh_device, ccl_manager)

    # Input: spatial tensor fractured on SP (dim 2) and TP (dim 3)
    # Shape: [1, B, N_local, D_local]
    batch_size = 1
    local_dim = WAN_DIM // tp_factor
    input_torch = torch.randn((1, batch_size, seq_len_local, local_dim), dtype=torch.float32)
    tt_spatial = bf16_tensor(input_torch, device=mesh_device)

    def run_spatial_qkv(spatial):
        # AllGather on TP axis to get full D dimension
        if tp_factor > 1:
            spatial_gathered = ccl_manager.all_gather_persistent_buffer(spatial, dim=3, mesh_axis=TP_AXIS)
        else:
            spatial_gathered = spatial

        # Q, K, V projections
        q = to_q(spatial_gathered)
        k = to_k(spatial_gathered)
        v = to_v(spatial_gathered)
        return q, k, v

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = run_spatial_qkv(tt_spatial)
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = run_spatial_qkv(tt_spatial)
    ttnn.synchronize_device(mesh_device)


@MESH_DEVICE_PARAMS
@CONFIG_WITH_SEQ_LEN_PARAMS
def test_run_ring_attention(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run ring attention (ring_joint_scaled_dot_product_attention) for spatial self-attention.
    Profiled via test_ring_attention_perf.
    """
    tp_factor = MESH_SHAPE[TP_AXIS]
    sp_factor = MESH_SHAPE[SP_AXIS]
    n_local_heads = WAN_NUM_HEADS // tp_factor
    seq_len_local = seq_len // sp_factor

    logger.info(f"Running ring attention: {config_name}, seq_len={seq_len}")

    ccl_manager = create_ccl_manager(mesh_device)

    # Get SDPA configs
    ring_sdpa_program_config, sdpa_compute_kernel_config, sdpa_worker_grid = get_ring_sdpa_configs(
        mesh_device, config_name
    )

    # Create Q, K, V tensors in BHNE format (after create_heads)
    # Shape: [B, H, N_local, E] where B=1, H=local_heads, N=seq_len_local, E=head_dim
    batch_size = 1
    q_shape = (batch_size, n_local_heads, seq_len_local, WAN_HEAD_DIM)
    k_shape = (batch_size, n_local_heads, seq_len_local, WAN_HEAD_DIM)
    v_shape = (batch_size, n_local_heads, seq_len_local, WAN_HEAD_DIM)

    torch.manual_seed(0)
    q_torch = torch.randn(q_shape, dtype=torch.float32)
    k_torch = torch.randn(k_shape, dtype=torch.float32)
    v_torch = torch.randn(v_shape, dtype=torch.float32)

    tt_q = bf16_tensor(q_torch, device=mesh_device)
    tt_k = bf16_tensor(k_torch, device=mesh_device)
    tt_v = bf16_tensor(v_torch, device=mesh_device)

    # Dummy joint inputs (empty tensors for pure self-attention)
    dummy_joint = bf16_tensor(torch.zeros((batch_size, n_local_heads, 0, WAN_HEAD_DIM)), device=mesh_device)

    def run_ring_attention():
        spatial_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(tt_k.shape, 2, SP_AXIS),
            persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(tt_v.shape, 2, SP_AXIS),
            joint_strategy="rear",
            logical_n=seq_len,
            program_config=ring_sdpa_program_config,
            compute_kernel_config=sdpa_compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(SP_AXIS),
            num_links=ccl_manager.num_links,
            cluster_axis=SP_AXIS,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(0, sdpa_worker_grid[1]),
        )
        return spatial_out

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = run_ring_attention()
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = run_ring_attention()
    ttnn.synchronize_device(mesh_device)


@MESH_DEVICE_PARAMS
@CONFIG_WITH_SEQ_LEN_PARAMS
def test_run_spatial_dense_single(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run spatial dense out / unfused Q proj: AllGather on TP axis followed by single matmul.
    This pattern occurs 3 times: cross-attn Q proj, self-attn dense out, cross-attn dense out.
    Profiled via test_spatial_dense_single_perf.
    """
    tp_factor = MESH_SHAPE[TP_AXIS]
    sp_factor = MESH_SHAPE[SP_AXIS]
    seq_len_local = seq_len // sp_factor

    logger.info(f"Running spatial dense single: {config_name}, seq_len={seq_len}")

    ccl_manager = create_ccl_manager(mesh_device)

    # Create single projection layer (ColParallelLinear)
    projection = create_col_parallel_linear(mesh_device, ccl_manager)

    # Input: spatial tensor fractured on SP (dim 2) and TP (dim 3)
    # Shape: [1, B, N_local, D_local]
    batch_size = 1
    local_dim = WAN_DIM // tp_factor
    input_torch = torch.randn((1, batch_size, seq_len_local, local_dim), dtype=torch.float32)
    tt_spatial = bf16_tensor(input_torch, device=mesh_device)

    def run_spatial_dense_single(spatial):
        # AllGather on TP axis to get full D dimension
        if tp_factor > 1:
            spatial_gathered = ccl_manager.all_gather_persistent_buffer(spatial, dim=3, mesh_axis=TP_AXIS)
        else:
            spatial_gathered = spatial

        # Single projection
        output = projection(spatial_gathered)
        return output

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = run_spatial_dense_single(tt_spatial)
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = run_spatial_dense_single(tt_spatial)
    ttnn.synchronize_device(mesh_device)


@MESH_DEVICE_PARAMS
@CONFIG_ONLY_PARAMS
def test_run_prompt_kv(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    reset_seeds,
) -> None:
    """
    Run prompt KV projection: K and V projections on prompt.
    Profiled via test_prompt_kv_perf.
    """
    logger.info(f"Running prompt KV: {config_name}")

    ccl_manager = create_ccl_manager(mesh_device)

    # Create K, V projection layers (ColParallelLinear)
    to_k = create_col_parallel_linear(mesh_device, ccl_manager)
    to_v = create_col_parallel_linear(mesh_device, ccl_manager)

    # Input: prompt tensor replicated across all devices
    # Shape: [1, B, L, D]
    batch_size = 1
    input_torch = torch.randn((1, batch_size, PROMPT_SEQ_LEN, WAN_DIM), dtype=torch.float32)
    tt_prompt = bf16_tensor(input_torch, device=mesh_device)

    def run_prompt_kv(prompt):
        k = to_k(prompt)
        v = to_v(prompt)
        return k, v

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = run_prompt_kv(tt_prompt)
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = run_prompt_kv(tt_prompt)
    ttnn.synchronize_device(mesh_device)


@MESH_DEVICE_PARAMS
@CONFIG_WITH_SEQ_LEN_PARAMS
def test_run_cross_attention(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run cross attention: scaled_dot_product_attention of spatial Q against prompt KV.
    Profiled via test_cross_attention_perf.
    """
    tp_factor = MESH_SHAPE[TP_AXIS]
    sp_factor = MESH_SHAPE[SP_AXIS]
    n_local_heads = WAN_NUM_HEADS // tp_factor
    seq_len_local = seq_len // sp_factor

    logger.info(f"Running cross attention: {config_name}, seq_len={seq_len}")

    # Create Q, K, V tensors in BHNE format
    # Q: spatial sequence (fractured on SP), K/V: prompt sequence (replicated on SP)
    batch_size = 1
    q_shape = (batch_size, n_local_heads, seq_len_local, WAN_HEAD_DIM)
    k_shape = (batch_size, n_local_heads, PROMPT_SEQ_LEN, WAN_HEAD_DIM)
    v_shape = (batch_size, n_local_heads, PROMPT_SEQ_LEN, WAN_HEAD_DIM)

    torch.manual_seed(0)
    q_torch = torch.randn(q_shape, dtype=torch.float32)
    k_torch = torch.randn(k_shape, dtype=torch.float32)
    v_torch = torch.randn(v_shape, dtype=torch.float32)

    tt_q = bf16_tensor(q_torch, device=mesh_device)
    tt_k = bf16_tensor(k_torch, device=mesh_device)
    tt_v = bf16_tensor(v_torch, device=mesh_device)

    # SDPA config for cross attention (no ring attention needed since prompt is replicated)
    full_grid = mesh_device.compute_with_storage_grid_size()
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=full_grid,
        q_chunk_size=256,
        k_chunk_size=256,
        exp_approx_mode=False,
    )

    sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )

    def run_cross_attention():
        output = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=False,
            program_config=sdpa_program_config,
            compute_kernel_config=sdpa_compute_kernel_config,
        )
        return output

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = run_cross_attention()
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = run_cross_attention()
    ttnn.synchronize_device(mesh_device)


@MESH_DEVICE_PARAMS
@CONFIG_WITH_SEQ_LEN_PARAMS
def test_run_spatial_ff1(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run spatial FF1: AllGather on TP axis followed by ColParallelLinear (with activation).
    Profiled via test_spatial_ff1_perf.
    """
    tp_factor = MESH_SHAPE[TP_AXIS]
    sp_factor = MESH_SHAPE[SP_AXIS]
    seq_len_local = seq_len // sp_factor

    logger.info(f"Running spatial FF1: {config_name}, seq_len={seq_len}")

    ccl_manager = create_ccl_manager(mesh_device)

    # Create FF1 layer (ColParallelLinear with activation)
    # FF1: dim -> ffn_dim with activation
    ff1 = ColParallelLinear(
        WAN_DIM,
        WAN_FFN_DIM,
        bias=True,
        activation_fn="gelu",
        mesh_device=mesh_device,
        mesh_axis=TP_AXIS,
        ccl_manager=ccl_manager,
    )

    # Load dummy weights
    dummy_weight = torch.randn(WAN_FFN_DIM, WAN_DIM, dtype=torch.float32)
    dummy_bias = torch.randn(WAN_FFN_DIM, dtype=torch.float32)
    dummy_state_dict = {"weight": dummy_weight, "bias": dummy_bias}
    ff1.load_state_dict(dummy_state_dict)

    # Input: spatial tensor fractured on SP (dim 2) and TP (dim 3)
    # Shape: [1, B, N_local, D_local]
    batch_size = 1
    local_dim = WAN_DIM // tp_factor
    input_torch = torch.randn((1, batch_size, seq_len_local, local_dim), dtype=torch.float32)
    tt_spatial = bf16_tensor(input_torch, device=mesh_device)

    def run_spatial_ff1(spatial):
        # AllGather on TP axis to get full D dimension
        if tp_factor > 1:
            spatial_gathered = ccl_manager.all_gather_persistent_buffer(spatial, dim=3, mesh_axis=TP_AXIS)
        else:
            spatial_gathered = spatial

        # FF1 projection with activation
        output = ff1(spatial_gathered)
        return output

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = run_spatial_ff1(tt_spatial)
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = run_spatial_ff1(tt_spatial)
    ttnn.synchronize_device(mesh_device)


@MESH_DEVICE_PARAMS
@CONFIG_WITH_SEQ_LEN_PARAMS
def test_run_spatial_ff2(
    mesh_device: ttnn.MeshDevice,
    config_name: str,
    seq_len: int,
    reset_seeds,
) -> None:
    """
    Run spatial FF2: RowParallelLinear (matmul + ReduceScatter on TP axis).
    Profiled via test_spatial_ff2_perf.
    """
    tp_factor = MESH_SHAPE[TP_AXIS]
    sp_factor = MESH_SHAPE[SP_AXIS]
    seq_len_local = seq_len // sp_factor

    logger.info(f"Running spatial FF2: {config_name}, seq_len={seq_len}")

    ccl_manager = create_ccl_manager(mesh_device)

    # Create FF2 layer (RowParallelLinear)
    # FF2: ffn_dim -> dim with ReduceScatter
    ff2 = RowParallelLinear(
        WAN_FFN_DIM,
        WAN_DIM,
        bias=True,
        mesh_device=mesh_device,
        mesh_axis=TP_AXIS,
        ccl_manager=ccl_manager,
    )

    # Load dummy weights
    dummy_weight = torch.randn(WAN_DIM, WAN_FFN_DIM, dtype=torch.float32)
    dummy_bias = torch.randn(WAN_DIM, dtype=torch.float32)
    dummy_state_dict = {"weight": dummy_weight, "bias": dummy_bias}
    ff2.load_state_dict(dummy_state_dict)

    # Input: FF1 output, fractured on SP (dim 2) and TP (dim 3)
    # Shape: [1, B, N_local, FFN_DIM_local]
    batch_size = 1
    local_ffn_dim = WAN_FFN_DIM // tp_factor
    input_torch = torch.randn((1, batch_size, seq_len_local, local_ffn_dim), dtype=torch.float32)
    tt_ff1_output = bf16_tensor(input_torch, device=mesh_device)

    def run_spatial_ff2(ff1_output):
        # RowParallelLinear: matmul + ReduceScatter on TP axis
        output = ff2(ff1_output)
        return output

    # Warmup
    for _ in range(NUM_WARMUP_ITERS):
        _ = run_spatial_ff2(tt_ff1_output)
    ttnn.synchronize_device(mesh_device)

    # Measurement
    for _ in range(NUM_MEASUREMENT_ITERS):
        _ = run_spatial_ff2(tt_ff1_output)
    ttnn.synchronize_device(mesh_device)


# =============================================================================
# Op group definitions
# =============================================================================

# Spatial layernorm: DistributedLayerNorm with dynamic affine parameters
SPATIAL_LAYERNORM_OPS = [
    ("PreAllGatherDeviceOperation", "pre_allgather"),
    ("AllGatherAsyncDeviceOperation", "all_gather"),
    ("PostAllGatherDeviceOperation", "post_allgather"),
]

# RMSNorm ops (used for both spatial and prompt RMSNorm)
RMSNORM_OPS = [
    ("FusedRMSNormPreAllGatherDeviceOperation", "pre_allgather"),
    ("AllGatherAsyncDeviceOperation", "all_gather"),
    ("FusedRMSNormPostAllGatherDeviceOperation", "post_allgather"),
]

# Spatial QKV: AllGather + 3x matmul (Q, K, V projections)
SPATIAL_QKV_OPS = [
    ("AllGatherAsyncDeviceOperation", "all_gather"),
    ("MinimalMatmulDeviceOperation", "matmul", 3),  # Q + K + V matmuls
]

# Ring Attention: ring_joint_scaled_dot_product_attention
RING_ATTENTION_OPS = [
    ("RingJointSDPADeviceOperation", "ring_sdpa"),
]

# Spatial dense out / unfused Q proj: AllGather + single matmul
# This pattern occurs 3x: cross-attn Q proj, self-attn dense out, cross-attn dense out
SPATIAL_DENSE_SINGLE_OPS = [
    ("AllGatherAsyncDeviceOperation", "all_gather"),
    ("MinimalMatmulDeviceOperation", "matmul"),
]

# Prompt KV projection: 2x matmul (K, V projections on prompt)
PROMPT_KV_OPS = [
    ("MinimalMatmulDeviceOperation", "matmul", 2),  # K + V matmuls
]

# Cross Attention: scaled_dot_product_attention (spatial Q x prompt KV)
CROSS_ATTENTION_OPS = [
    ("SDPAOperation", "sdpa"),
]

# Spatial FF1: AllGather + single matmul with activation
SPATIAL_FF1_OPS = [
    ("AllGatherAsyncDeviceOperation", "all_gather"),
    ("MinimalMatmulDeviceOperation", "matmul"),
]

# Spatial FF2: matmul + ReduceScatter
SPATIAL_FF2_OPS = [
    ("MinimalMatmulDeviceOperation", "matmul"),
    ("ReduceScatterMinimalAsyncDeviceOperation", "reduce_scatter"),
]


# =============================================================================
# Performance tests (run with Tracy profiler)
# =============================================================================


@CONFIG_WITH_SEQ_LEN_PARAMS
def test_spatial_layernorm_perf(config_name: str, seq_len: int) -> None:
    """Measure device performance for spatial layernorm (DistributedLayerNorm)."""
    run_perf_test("test_run_spatial_layernorm", config_name, seq_len, "SPATIAL LAYERNORM", SPATIAL_LAYERNORM_OPS)


@CONFIG_WITH_SEQ_LEN_PARAMS
def test_spatial_rmsnorm_perf(config_name: str, seq_len: int) -> None:
    """Measure device performance for spatial RMSNorm with fused RoPE."""
    run_perf_test("test_run_spatial_rmsnorm", config_name, seq_len, "SPATIAL RMSNORM", RMSNORM_OPS)


@CONFIG_ONLY_PARAMS
def test_prompt_rmsnorm_perf(config_name: str) -> None:
    """Measure device performance for prompt RMSNorm without RoPE."""
    run_perf_test("test_run_prompt_rmsnorm", config_name, PROMPT_SEQ_LEN, "PROMPT RMSNORM", RMSNORM_OPS)


@CONFIG_WITH_SEQ_LEN_PARAMS
def test_spatial_qkv_perf(config_name: str, seq_len: int) -> None:
    """Measure device performance for spatial QKV (AllGather + Q/K/V projections)."""
    run_perf_test("test_run_spatial_qkv", config_name, seq_len, "SPATIAL QKV", SPATIAL_QKV_OPS)


@CONFIG_WITH_SEQ_LEN_PARAMS
def test_ring_attention_perf(config_name: str, seq_len: int) -> None:
    """Measure device performance for ring attention (spatial self-attention)."""
    run_perf_test("test_run_ring_attention", config_name, seq_len, "RING ATTENTION", RING_ATTENTION_OPS)


@CONFIG_WITH_SEQ_LEN_PARAMS
def test_spatial_dense_single_perf(config_name: str, seq_len: int) -> None:
    """Measure device performance for spatial dense out / unfused Q proj (AllGather + matmul)."""
    run_perf_test(
        "test_run_spatial_dense_single", config_name, seq_len, "SPATIAL DENSE SINGLE", SPATIAL_DENSE_SINGLE_OPS
    )


@CONFIG_ONLY_PARAMS
def test_prompt_kv_perf(config_name: str) -> None:
    """Measure device performance for prompt KV projection (K + V matmuls)."""
    run_perf_test("test_run_prompt_kv", config_name, PROMPT_SEQ_LEN, "PROMPT KV", PROMPT_KV_OPS)


@CONFIG_WITH_SEQ_LEN_PARAMS
def test_cross_attention_perf(config_name: str, seq_len: int) -> None:
    """Measure device performance for cross attention (spatial Q x prompt KV)."""
    run_perf_test("test_run_cross_attention", config_name, seq_len, "CROSS ATTENTION", CROSS_ATTENTION_OPS)


@CONFIG_WITH_SEQ_LEN_PARAMS
def test_spatial_ff1_perf(config_name: str, seq_len: int) -> None:
    """Measure device performance for spatial FF1 (AllGather + matmul with activation)."""
    run_perf_test("test_run_spatial_ff1", config_name, seq_len, "SPATIAL FF1", SPATIAL_FF1_OPS)


@CONFIG_WITH_SEQ_LEN_PARAMS
def test_spatial_ff2_perf(config_name: str, seq_len: int) -> None:
    """Measure device performance for spatial FF2 (matmul + ReduceScatter)."""
    run_perf_test("test_run_spatial_ff2", config_name, seq_len, "SPATIAL FF2", SPATIAL_FF2_OPS)
