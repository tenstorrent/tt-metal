# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for gpt_oss_experts_mlp.

This fused op tests the expert MLP computation with batched matmul:
1. Gate/Up projection (w1/w3) with batched matmul + bias (or fused w1_w3 single matmul)
2. SwiGLU activation: (up_clamped + 1) * (gate_clamped * sigmoid(gate_clamped * alpha))
3. Down projection (w2) with batched matmul + bias
4. Reshape and layout conversion for output

This is a decode and prefill fused op (same code path for both modes).
Does not contain CCL ops, so single device test is applicable.
"""

import json
import math
import os
from collections import defaultdict

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.demos.gpt_oss.tests.test_factory import TestFactory
from models.demos.gpt_oss.tt.experts_throughput.config import ThroughputExpertConfig, ThroughputProgramConfig
from models.demos.gpt_oss.tt.experts_throughput.decode import expert_mlp_forward
from models.demos.gpt_oss.tt.experts_throughput.weights import ThroughputExpertWeights, load_throughput_expert_weights
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

# ==============================================================================
# Constants
# ==============================================================================
DEVICE_PERF_ENV_VAR = "GPT_OSS_EXPERTS_MLP_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1

# TODO: Set device perf targets based on measured baselines
DEVICE_PERF_TARGETS_US = {}

# ==============================================================================
# Performance Measurement Methodology Notes
# ==============================================================================
# This test measures two key device performance metrics:
#
# 1. KERNEL DURATION: The actual time the device spends executing compute kernels.
#    This is the "useful work" metric and represents the theoretical minimum time
#    for the fused op sequence.
#
# 2. OP-TO-OP LATENCY: The time between when one op completes and the next starts.
#    This includes host-side overhead, command queue submission, and synchronization.
#
# IMPORTANT: The op-to-op latency measured here is INFLATED compared to production!
#
# In the device perf measurement loop, we call `ttnn.synchronize_device()` after
# each iteration to ensure accurate per-iteration timing. This synchronization:
#   - Waits for all device operations to complete
#   - Returns control to the host
#   - Adds significant latency between iterations
#
# In production inference with proper tracing, traces are pipelined without
# per-iteration sync, so the device executes back-to-back with near-zero op-to-op
# latency. The KERNEL DURATION is the metric that best represents production perf.
# ==============================================================================


# ==============================================================================
# PyTorch Reference Implementations
# ==============================================================================
def _apply_swiglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """PyTorch reference implementation for SwiGLU activation.

    Implements: (up_clamped + 1) * (gate_clamped * sigmoid(gate_clamped * alpha))

    Args:
        gate: Gate projection output
        up: Up projection output
        alpha: Scaling factor for sigmoid
        limit: Clamping limit for swiglu

    Returns:
        Activated tensor
    """
    # Clamp gate (max only)
    gate_clamped = torch.clamp(gate, max=limit)
    # Clamp up (both min and max)
    up_clamped = torch.clamp(up, min=-limit, max=limit)
    # Compute gate_alpha = gate * alpha
    gate_alpha = gate_clamped * alpha
    # Compute gate_sigmoid = sigmoid(gate_alpha)
    gate_sigmoid = torch.sigmoid(gate_alpha)
    # Compute glu = gate * gate_sigmoid
    glu = gate_clamped * gate_sigmoid
    # Add 1 to up: up = up + 1
    up_plus_one = up_clamped + 1.0
    # Multiply: result = up * glu
    result = up_plus_one * glu
    return result


def gpt_oss_experts_mlp_reference(
    post_dispatch: torch.Tensor,
    w1: torch.Tensor,
    w1_bias: torch.Tensor,
    w3: torch.Tensor,
    w3_bias: torch.Tensor,
    w2: torch.Tensor,
    config: ThroughputExpertConfig,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """PyTorch reference implementation for expert_mlp_forward.

    Computes expert MLP forward pass:
    output = down((up + 1) * (gate * sigmoid(gate * alpha)))

    Note: This is a dense reference that computes all experts for all tokens,
    unlike the TTNN version which uses sparse matmul. The reference should
    produce equivalent results when compared on active (token, expert) pairs.

    Args:
        post_dispatch: Input tensor [1, 1, B*S, H] in tile-compatible format
        w1: Gate projection weights [num_experts_per_device, H, I]
        w1_bias: Gate projection bias [num_experts_per_device, 1, I]
        w3: Up projection weights [num_experts_per_device, H, I]
        w3_bias: Up projection bias [num_experts_per_device, 1, I]
        w2: Down projection weights [num_experts_per_device, I, H]
        config: ThroughputExpertConfig
        batch_size: Global batch size
        seq_len: Sequence length

    Returns:
        Expert output tensor [num_experts_per_device, B, S, H]
    """
    total_tokens = batch_size * seq_len
    num_experts = w1.shape[0]
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    # Reshape input: [1, 1, B*S, H] -> [B*S, H]
    x = post_dispatch.reshape(total_tokens, hidden_size)

    # Expand for all experts: [B*S, H] -> [num_experts, B*S, H]
    x_expanded = x.unsqueeze(0).expand(num_experts, -1, -1)

    # Gate projection: [num_experts, B*S, H] @ [num_experts, H, I] -> [num_experts, B*S, I]
    gate = torch.bmm(x_expanded, w1)
    # Add bias: [num_experts, B*S, I] + [num_experts, 1, I]
    gate = gate + w1_bias

    # Up projection: [num_experts, B*S, H] @ [num_experts, H, I] -> [num_experts, B*S, I]
    up = torch.bmm(x_expanded, w3)
    # Add bias
    up = up + w3_bias

    # SwiGLU activation
    activated = _apply_swiglu_reference(gate, up, config.alpha, config.swiglu_limit)

    # Down projection: [num_experts, B*S, I] @ [num_experts, I, H] -> [num_experts, B*S, H]
    output = torch.bmm(activated, w2)

    # Reshape to [num_experts, B, S, H]
    output = output.reshape(num_experts, batch_size, seq_len, hidden_size)

    return output


# ==============================================================================
# TTNN Implementation
# ==============================================================================
def gpt_oss_experts_mlp_ttnn(
    post_dispatch: ttnn.Tensor,
    weights,
    config: ThroughputExpertConfig,
    program_config: ThroughputProgramConfig,
    memory_config: ttnn.MemoryConfig,
    total_tokens: int,
) -> ttnn.Tensor:
    """TTNN implementation for expert_mlp_forward.

    This is exactly the expert_mlp_forward function from decode.py.

    Args:
        post_dispatch: Input tensor [1, num_experts_per_device, total_tokens, H] in TILE layout
        weights: ThroughputExpertWeights with w2, w2_bias, and either fused or unfused gate/up weights
        config: ThroughputExpertConfig
        program_config: ThroughputProgramConfig
        memory_config: Output memory configuration
        total_tokens: Total number of tokens (B*S)

    Returns:
        Expert output tensor [experts_per_device, 1, total_tokens, H] in ROW_MAJOR layout
    """
    return expert_mlp_forward(
        experts_input=post_dispatch,
        weights=weights,
        config=config,
        memory_config=memory_config,
        program_config=program_config,
        total_tokens=total_tokens,
    )


# ==============================================================================
# Helper Functions
# ==============================================================================
def _compare_with_reference(
    tt_output: torch.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    name: str = "",
) -> tuple[bool, float]:
    """Compare TT output with reference, returning pass status and PCC."""
    passing, pcc = comp_pcc(ref_output.float(), tt_output.float(), expected_pcc)
    logger.info(f"PCC {name}: {pcc}")
    return passing, pcc


def _measure_perf_us(
    mesh_device: ttnn.MeshDevice,
    op_fn,
    warmup_iters: int,
    measure_iters: int,
    trace_mode: bool = False,
) -> float:
    """Measure performance in microseconds."""
    ttnn.synchronize_device(mesh_device)

    # Non-trace mode
    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("fused_op_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("fused_op_perf", PERF_CNT=measure_iters)
    return profiler.get("fused_op_perf") * 1e6


def _merge_device_rows_for_perf(df: pd.DataFrame) -> pd.DataFrame:
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


def _collect_device_perf(
    command: str, subdir: str, warmup_iters: int, use_signposts: bool = False
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Collect device performance metrics using Tracy profiler."""
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
    df = _merge_device_rows_for_perf(df)

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


def _create_reference_weights(config, num_experts):
    """Create random weights for both reference and TTNN models."""
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    reference_layer = GptOssDecoderLayer(config, layer_idx=0)
    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, 1)

    reference_experts = reference_layer.mlp.experts.eval()

    # Extract weights from reference model
    state_dict = {
        "gate_up_proj": reference_experts.gate_up_proj.data.clone(),
        "gate_up_proj_bias": reference_experts.gate_up_proj_bias.data.clone(),
        "down_proj": reference_experts.down_proj.data.clone(),
        "down_proj_bias": reference_experts.down_proj_bias.data.clone(),
    }

    return state_dict


# ==============================================================================
# Test Implementation
# ==============================================================================
def _run_experts_mlp_test(
    mesh_device: ttnn.MeshDevice,
    config,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    intermediate_size: int,
    expected_pcc: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    step_prefix: str,
):
    """Run the expert_mlp_forward fused op test."""
    mesh_shape = mesh_device.shape
    num_devices = mesh_device.get_num_devices()

    # Determine batch per device for row sharding
    if batch_size > 32:
        is_row_sharded = True
        assert batch_size % mesh_shape[0] == 0, "Batch size must be divisible by mesh rows"
        batch_size_per_device = batch_size // mesh_shape[0]
    else:
        is_row_sharded = False
        batch_size_per_device = batch_size

    # Create ThroughputExpertConfig
    throughput_config = ThroughputExpertConfig(
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=num_devices,
    )

    # Create weights
    state_dict = _create_reference_weights(config, num_experts)
    weights = load_throughput_expert_weights(
        mesh_device=mesh_device,
        config=throughput_config,
        state_dict=state_dict,
        weight_dtype=ttnn.bfloat16,
    )

    # Extract reference weights from state_dict for PyTorch computation
    # HuggingFace format: gate_up_proj [num_experts, hidden, 2*intermediate] (interleaved)
    gate_up = state_dict["gate_up_proj"]
    w1_ref = gate_up[..., ::2]  # [num_experts, hidden, intermediate]
    w3_ref = gate_up[..., 1::2]  # [num_experts, hidden, intermediate]
    gate_up_bias = state_dict["gate_up_proj_bias"]
    w1_bias_ref = gate_up_bias[..., ::2].unsqueeze(1)  # [num_experts, 1, intermediate]
    w3_bias_ref = gate_up_bias[..., 1::2].unsqueeze(1)  # [num_experts, 1, intermediate]
    w2_ref = state_dict["down_proj"]  # [num_experts, intermediate, hidden]

    # Create input tensor (post_dispatch output)
    # Shape: [1, 1, B*S, H]
    total_tokens = batch_size * seq_len
    num_experts_per_device = throughput_config.num_experts_per_device
    post_dispatch_torch = torch.randn(1, 1, total_tokens, hidden_size, dtype=torch.bfloat16)

    # Reference computation - use only first device's experts
    ref_output = gpt_oss_experts_mlp_reference(
        post_dispatch=post_dispatch_torch.float(),
        w1=w1_ref[:num_experts_per_device].float(),
        w1_bias=w1_bias_ref[:num_experts_per_device].float(),
        w3=w3_ref[:num_experts_per_device].float(),
        w3_bias=w3_bias_ref[:num_experts_per_device].float(),
        w2=w2_ref[:num_experts_per_device].float(),
        config=throughput_config,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    # Convert to TTNN tensors
    tt_post_dispatch = ttnn.from_torch(
        post_dispatch_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Repeat input across expert dimension for batched matmul:
    # [1, 1, total_tokens, H] -> [1, num_experts_per_device, total_tokens, H]
    tt_post_dispatch = ttnn.repeat(tt_post_dispatch, ttnn.Shape((1, num_experts_per_device, 1, 1)))

    # Create program config
    program_config = ThroughputProgramConfig()
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Run TTNN implementation
    tt_output = gpt_oss_experts_mlp_ttnn(
        post_dispatch=tt_post_dispatch,
        weights=weights,
        config=throughput_config,
        program_config=program_config,
        memory_config=memory_config,
        total_tokens=total_tokens,
    )

    # Convert output to torch - get first device output
    tt_output_tensors = ttnn.get_device_tensors(tt_output)
    tt_output_torch = ttnn.to_torch(tt_output_tensors[0])
    # Reshape to match reference: [experts_per_device, B, S, H]
    tt_output_torch = tt_output_torch.reshape(num_experts_per_device, batch_size, seq_len, hidden_size)

    # Compare with reference
    passing, pcc = _compare_with_reference(
        tt_output_torch,
        ref_output,
        expected_pcc,
        "experts_mlp",
    )
    assert passing, f"experts_mlp test failed. PCC: {pcc} < {expected_pcc}"

    # Re-create input tensor for perf measurement since expert_mlp_forward deallocates it
    tt_post_dispatch = ttnn.from_torch(
        post_dispatch_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_post_dispatch = ttnn.repeat(tt_post_dispatch, ttnn.Shape((1, num_experts_per_device, 1, 1)))

    # Performance measurement
    # expert_mlp_forward deallocates experts_input internally, so we must
    # clone the input tensor for each iteration to avoid "Buffer not allocated" errors.
    def op_fn():
        return gpt_oss_experts_mlp_ttnn(
            post_dispatch=ttnn.clone(tt_post_dispatch),
            weights=weights,
            config=throughput_config,
            program_config=program_config,
            memory_config=memory_config,
            total_tokens=total_tokens,
        )

    # Device performance measurement mode
    if os.getenv(DEVICE_PERF_ENV_VAR) is not None:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        from tracy import signpost

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        signpost("start")
        for _ in range(DEVICE_PERF_ITERS):
            # NOTE: The sync after each iteration inflates op-to-op latency but is required
            # for accurate per-iteration measurement. Production would use traced execution
            # with pipelined iterations. See module docstring for details.
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)
        signpost("stop")
        return pcc

    # Standard e2e performance measurement
    if not trace_mode or program_cache_enabled:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        perf_profiler.start("run")
        perf_profiler.start(step_name)
        perf_us = _measure_perf_us(
            mesh_device,
            op_fn,
            PERF_WARMUP_ITERS,
            PERF_MEASURE_ITERS,
            trace_mode=False,
        )
        logger.info(f"Perf avg: {perf_us:.3f} us over {PERF_MEASURE_ITERS} iters (warmup {PERF_WARMUP_ITERS})")
        perf_profiler.end(step_name)
        perf_profiler.end("run")

        benchmark_data.add_measurement(
            perf_profiler,
            0,
            step_name,
            f"{step_name}-avg_us",
            perf_us,
            step_warm_up_num_iterations=PERF_WARMUP_ITERS,
            target=expected_perf_us if expected_perf_us > 0 and program_cache_enabled else None,
        )
        benchmark_data.save_partial_run_json(
            perf_profiler,
            run_type="gpt_oss_fused_ops",
            ml_model_name="gpt-oss",
            batch_size=batch_size,
            input_sequence_length=seq_len,
        )

        if expected_perf_us > 0 and program_cache_enabled:
            perf_margin = 0.2
            assert perf_us <= expected_perf_us * (
                1 + perf_margin
            ), f"Perf regression: {perf_us:.3f}us exceeds expected {expected_perf_us:.3f}us"
        elif expected_perf_us == 0 and program_cache_enabled:
            logger.warning("TODO: Set expected_perf_us using a measured baseline.")

    return pcc


# ==============================================================================
# Pytest Test Functions
# ==============================================================================
@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_perf_us",
    [
        # Decode mode (seq_len=1) - Measured PCC: 0.9983
        ("decode", 1, 0.998, 0.0),  # TODO: Set expected_perf_us based on measured baseline
        # Prefill modes
        ("prefill", 128, 0.998, 0.0),
    ],
)
@pytest.mark.parametrize("use_real_weights", [False], ids=["random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False], ids=["eager"])  # Trace mode disabled for MLP
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 8)],
    ids=["mesh_4x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 30000000,
        }
    ],
    indirect=True,
)
def test_gpt_oss_experts_mlp(
    mode,
    seq_len,
    expected_pcc,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Test the gpt_oss_experts_mlp fused op.

    This tests the expert MLP computation:
    - Gate/Up projection with batched matmul + bias (fused or unfused)
    - SwiGLU activation
    - Down projection (w2) with batched matmul + bias
    - Reshape and layout conversion for output
    """
    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # Get HF config
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Use config values for dimensions
    hidden_size = config.hidden_size
    num_experts = config.num_local_experts
    num_experts_per_tok = config.num_experts_per_tok
    intermediate_size = config.intermediate_size

    # Batch size based on mode
    batch_size = 128 if mode == "decode" else 1

    pcc = _run_experts_mlp_test(
        mesh_device,
        config,
        batch_size,
        seq_len,
        hidden_size,
        num_experts,
        num_experts_per_tok,
        intermediate_size,
        expected_pcc,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        f"gpt_oss_experts_mlp_{mode}_seq{seq_len}",
    )

    logger.info(f"Test passed with PCC: {pcc}")


# ==============================================================================
# Single Device Test
# ==============================================================================
@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_perf_us",
    [
        ("decode", 1, 0.998, 0.0),  # Measured PCC: 0.9983
    ],
)
@pytest.mark.parametrize("use_real_weights", [False], ids=["random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True], ids=["program_cache"])
@pytest.mark.parametrize("trace_mode", [False], ids=["eager"])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 8)],
    ids=["mesh_4x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 30000000,
        }
    ],
    indirect=True,
)
def test_gpt_oss_experts_mlp_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Single device test for gpt_oss_experts_mlp.

    This test can run on single device since expert_mlp_forward
    does not contain CCL ops (CCLs are in decode_forward wrapper).
    """
    # Create a 1x1 submesh to get a single device
    single_device_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 1)))

    # Get HF config
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    hidden_size = config.hidden_size
    num_experts = config.num_local_experts
    num_experts_per_tok = config.num_experts_per_tok
    intermediate_size = config.intermediate_size
    batch_size_per_device = 32  # Single device batch size
    num_devices = 1  # Single device

    # Create ThroughputExpertConfig for single device
    # Use unfused mode since we manually create separate w1/w3 weights below
    throughput_config = ThroughputExpertConfig(
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=num_devices,
        use_fused_gate_up=False,
    )

    # Create weights - for single device we use all experts
    state_dict = _create_reference_weights(config, num_experts)

    # Extract reference weights
    gate_up = state_dict["gate_up_proj"]
    w1_ref = gate_up[..., ::2]
    w3_ref = gate_up[..., 1::2]
    gate_up_bias = state_dict["gate_up_proj_bias"]
    w1_bias_ref = gate_up_bias[..., ::2].unsqueeze(1)
    w3_bias_ref = gate_up_bias[..., 1::2].unsqueeze(1)
    w2_ref = state_dict["down_proj"]

    # Create input tensor
    total_tokens = batch_size_per_device * seq_len
    num_experts_per_device = throughput_config.num_experts_per_device
    post_dispatch_torch = torch.randn(1, 1, total_tokens, hidden_size, dtype=torch.bfloat16)

    # Reference computation
    ref_output = gpt_oss_experts_mlp_reference(
        post_dispatch=post_dispatch_torch.float(),
        w1=w1_ref[:num_experts_per_device].float(),
        w1_bias=w1_bias_ref[:num_experts_per_device].float(),
        w3=w3_ref[:num_experts_per_device].float(),
        w3_bias=w3_bias_ref[:num_experts_per_device].float(),
        w2=w2_ref[:num_experts_per_device].float(),
        config=throughput_config,
        batch_size=batch_size_per_device,
        seq_len=seq_len,
    )

    # Create TTNN weights for single device submesh
    # Weights shape: [1, num_experts_per_device, H, I] for gate/up, [1, num_experts_per_device, I, H] for down
    w1_tt = ttnn.from_torch(
        w1_ref[:num_experts_per_device].unsqueeze(0),
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
    )
    w2_tt = ttnn.from_torch(
        w2_ref[:num_experts_per_device].unsqueeze(0),
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
    )
    w3_tt = ttnn.from_torch(
        w3_ref[:num_experts_per_device].unsqueeze(0),
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
    )
    # Bias shapes: [1, num_experts_per_device, 1, dim]
    w1_bias_tt = ttnn.from_torch(
        w1_bias_ref[:num_experts_per_device].unsqueeze(0),
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
    )
    w2_bias_tt = ttnn.from_torch(
        torch.zeros(1, num_experts_per_device, 1, hidden_size),
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
    )
    w3_bias_tt = ttnn.from_torch(
        w3_bias_ref[:num_experts_per_device].unsqueeze(0),
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
    )

    weights = ThroughputExpertWeights(
        w2=w2_tt, w2_bias=w2_bias_tt, w1=w1_tt, w3=w3_tt, w1_bias=w1_bias_tt, w3_bias=w3_bias_tt
    )

    # Convert inputs to TTNN
    tt_post_dispatch = ttnn.from_torch(
        post_dispatch_torch,
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
    )

    # Repeat input across expert dimension for batched matmul:
    # [1, 1, total_tokens, H] -> [1, num_experts_per_device, total_tokens, H]
    tt_post_dispatch = ttnn.repeat(tt_post_dispatch, ttnn.Shape((1, num_experts_per_device, 1, 1)))

    program_config = ThroughputProgramConfig()
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Run TTNN implementation
    tt_output = expert_mlp_forward(
        experts_input=tt_post_dispatch,
        weights=weights,
        config=throughput_config,
        memory_config=memory_config,
        program_config=program_config,
        total_tokens=total_tokens,
    )

    # Convert output to torch
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_output_torch = tt_output_torch.reshape(num_experts_per_device, batch_size_per_device, seq_len, hidden_size)

    # Compare with reference
    passing, pcc = _compare_with_reference(
        tt_output_torch,
        ref_output,
        expected_pcc,
        "experts_mlp_single_device",
    )
    assert passing, f"Single device experts_mlp test failed. PCC: {pcc} < {expected_pcc}"
    logger.info(f"Single device test passed with PCC: {pcc}")


# ==============================================================================
# Device Performance Test
# ==============================================================================
@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_gpt_oss_experts_mlp_device_perf(mode, seq_len):
    """Device performance test for gpt_oss_experts_mlp.

    This test measures device kernel duration and op-to-op latency using Tracy profiler.
    """
    # Batch size for decode_128 configuration
    batch_size = 128 if mode == "decode" else 1

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"gpt_oss_experts_mlp_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_experts_mlp.py"
    expr = f"program_cache and not no_program_cache and eager and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_gpt_oss_experts_mlp -k "{expr}"'

    perf_profiler.start("run")
    perf_profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = _collect_device_perf(
        command,
        subdir="gpt_oss_fused_ops_device_perf",
        warmup_iters=0,
        use_signposts=True,
    )
    os.environ.pop(DEVICE_PERF_ENV_VAR, None)
    perf_profiler.end(step_name)
    perf_profiler.end("run")

    assert op_stats, "No device perf stats captured."
    total_kernel_us = total_kernel_ns / 1000.0
    total_op_to_op_us = total_op_to_op_ns / 1000.0
    avg_kernel_us = total_kernel_us / DEVICE_PERF_ITERS
    avg_op_to_op_us = total_op_to_op_us / DEVICE_PERF_ITERS
    logger.info(f"Device perf per-op averages (ns): {json.dumps(op_stats, indent=2)}")
    logger.info(
        f"Device perf totals ({DEVICE_PERF_ITERS} iterations): kernel={total_kernel_us:.3f} us, op_to_op={total_op_to_op_us:.3f} us"
    )
    logger.info(f"Device perf per-iteration averages: kernel={avg_kernel_us:.3f} us, op_to_op={avg_op_to_op_us:.3f} us")
    assert total_kernel_ns > 0, "Total kernel duration must be positive."
    assert total_op_to_op_ns >= 0, "Total op-to-op latency must be non-negative."

    targets = DEVICE_PERF_TARGETS_US.get((mode, seq_len))
    if targets is None:
        logger.warning("No device perf targets configured; skipping perf assertions.")
    else:
        kernel_target_us = targets["kernel"]
        op_to_op_target_us = targets["op_to_op"]
        kernel_limit_us = kernel_target_us * (1 + DEVICE_PERF_MARGIN)
        op_to_op_limit_us = op_to_op_target_us * (1 + DEVICE_PERF_MARGIN)
        assert (
            total_kernel_us <= kernel_limit_us
        ), f"Kernel perf regression: {total_kernel_us:.3f}us exceeds {kernel_target_us:.3f}us (+{DEVICE_PERF_MARGIN:.0%})"
        assert (
            total_op_to_op_us <= op_to_op_limit_us
        ), f"Op-to-op perf regression: {total_op_to_op_us:.3f}us exceeds {op_to_op_target_us:.3f}us (+{DEVICE_PERF_MARGIN:.0%})"

    benchmark_data.add_measurement(
        perf_profiler,
        0,
        step_name,
        "total_kernel_duration_us",
        total_kernel_us,
    )
    benchmark_data.add_measurement(
        perf_profiler,
        0,
        step_name,
        "total_op_to_op_latency_us",
        total_op_to_op_us,
    )
    benchmark_data.add_measurement(
        perf_profiler,
        0,
        step_name,
        "avg_kernel_duration_us",
        avg_kernel_us,
    )
    benchmark_data.add_measurement(
        perf_profiler,
        0,
        step_name,
        "avg_op_to_op_latency_us",
        avg_op_to_op_us,
    )
    benchmark_data.save_partial_run_json(
        perf_profiler,
        run_type="gpt_oss_fused_ops_device_perf",
        ml_model_name="gpt-oss",
        batch_size=batch_size,
        input_sequence_length=seq_len,
    )
