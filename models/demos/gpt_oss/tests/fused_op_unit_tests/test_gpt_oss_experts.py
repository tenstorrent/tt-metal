# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for gpt_oss_experts (full decode_forward).

This fused op tests the entire throughput experts decode forward pass including:
1. Tensor preparation (reshape, typecast, layout conversion)
2. all_to_all_dispatch - Route tokens to expert devices (CCL)
3. moe_expert_token_remap - Create sparsity pattern
4. Expert computation - Gate/Up/Down projections with sparse matmul + SwiGLU
5. all_to_all_combine - Route expert outputs back (CCL)
6. Apply routing weights and reduce across experts
7. all_reduce - Aggregate across columns (CCL)

This is a decode-only fused op (seq_len=1).
Contains CCL ops so single device test is skipped.
"""

import itertools
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
from models.demos.gpt_oss.tt.experts_throughput.config import (
    AllToAllCombineConfig,
    AllToAllDispatchConfig,
    ThroughputExpertConfig,
    ThroughputProgramConfig,
    create_expert_mapping_tensors,
    create_remap_topk_mask,
)
from models.demos.gpt_oss.tt.experts_throughput.decode import decode_forward
from models.demos.gpt_oss.tt.experts_throughput.weights import load_throughput_expert_weights
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

DEVICE_PERF_ENV_VAR = "GPT_OSS_EXPERTS_DEVICE_PERF"
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


def gpt_oss_experts_reference(
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    reference_experts,
) -> torch.Tensor:
    """PyTorch reference implementation for gpt_oss_experts.

    Uses the HuggingFace GptOss experts module as the reference.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        router_indices: Expert indices per token [batch * seq_len, num_experts_per_tok]
        routing_weights: Routing weights (sparse) [batch * seq_len, num_experts]
        reference_experts: HuggingFace GptOss experts module

    Returns:
        Output tensor [batch, seq_len, hidden_size]
    """
    # Convert to float32 for reference computation (HuggingFace model uses float32 weights)
    hidden_states_fp32 = hidden_states.float()
    routing_weights_fp32 = routing_weights.float()

    with torch.no_grad():
        output = reference_experts(
            hidden_states_fp32,
            router_indices=router_indices,
            routing_weights=routing_weights_fp32,
        )
    return output


def gpt_oss_experts_ttnn(
    hidden_states: ttnn.Tensor,
    topk_expert_indices: ttnn.Tensor,
    topk_expert_weights: ttnn.Tensor,
    weights,
    config: ThroughputExpertConfig,
    expert_mapping_tensors: ttnn.Tensor,
    remap_topk_mask: ttnn.Tensor,
    dispatch_config: AllToAllDispatchConfig,
    combine_config: AllToAllCombineConfig,
    program_config: ThroughputProgramConfig,
    mesh_device,
) -> ttnn.Tensor:
    """TTNN implementation for gpt_oss_experts.

    This is the full decode_forward function from throughput experts.

    Args:
        hidden_states: Input tensor [batch_per_device, 1, seq_len, hidden_size]
        topk_expert_indices: Expert indices [batch_per_device, 1, seq_len, k]
        topk_expert_weights: Routing weights [batch_per_device, 1, seq_len, k]
        weights: ThroughputExpertWeights with w2, w2_bias, and either fused or unfused gate/up weights
        config: ThroughputExpertConfig
        expert_mapping_tensors: Device-to-expert mapping
        remap_topk_mask: Mask for expert remapping
        dispatch_config: AllToAllDispatchConfig
        combine_config: AllToAllCombineConfig
        program_config: ThroughputProgramConfig
        mesh_device: TTNN mesh device

    Returns:
        Output tensor [1, 1, batch_per_device * seq_len, hidden_size]
    """
    return decode_forward(
        hidden_states,
        topk_expert_indices,
        topk_expert_weights,
        weights,
        config,
        expert_mapping_tensors,
        remap_topk_mask,
        dispatch_config,
        combine_config,
        program_config,
        mesh_device,
    )


def _compare_with_reference(
    tt_output: torch.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    atol: float,
    rtol: float,
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

    if trace_mode:
        # Initial run to allocate tensors
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                output = op_fn()
                ttnn.deallocate(output)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            output = op_fn()
            ttnn.deallocate(output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("gpt_oss_experts_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("gpt_oss_experts_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("gpt_oss_experts_perf") * 1e6

    # Non-trace mode
    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("gpt_oss_experts_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("gpt_oss_experts_perf", PERF_CNT=measure_iters)
    return profiler.get("gpt_oss_experts_perf") * 1e6


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
            # Also track totals for proper per-iteration calculation
            "total_kernel_duration_ns": sum(kernel_vals),
            "total_op_to_op_latency_ns": sum(op_to_op_vals),
            "call_count": len(kernel_vals),
        }

    # Calculate total kernel/op-to-op time across ALL ops (not just averages per op type)
    # This gives accurate totals that match tt-perf-report stacked output
    total_kernel_ns = sum(entry["total_kernel_duration_ns"] for entry in op_stats.values())
    total_op_to_op_ns = sum(entry["total_op_to_op_latency_ns"] for entry in op_stats.values())
    return op_stats, total_kernel_ns, total_op_to_op_ns


def _create_reference_experts_and_weights(config, num_experts):
    """Create HuggingFace reference experts and extract weights for TTNN.

    Returns both the reference experts module and a state dict compatible with
    load_throughput_expert_weights.
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    reference_layer = GptOssDecoderLayer(config, layer_idx=0)
    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, 1)

    reference_experts = reference_layer.mlp.experts.eval()

    # Extract weights from reference model and convert to expected format
    # HuggingFace format: gate_up_proj [num_experts, hidden, 2*intermediate] (interleaved)
    # down_proj [num_experts, intermediate, hidden]
    state_dict = {
        "gate_up_proj": reference_experts.gate_up_proj.data.clone(),
        "gate_up_proj_bias": reference_experts.gate_up_proj_bias.data.clone(),
        "down_proj": reference_experts.down_proj.data.clone(),
        "down_proj_bias": reference_experts.down_proj_bias.data.clone(),
    }

    return reference_experts, state_dict


def _run_experts_test(
    mesh_device: ttnn.MeshDevice,
    config,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    intermediate_size: int,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    use_real_weights: bool,
    step_prefix: str,
):
    """Run the full experts fused op test."""

    # Determine batch per device and row sharding
    mesh_shape = mesh_device.shape
    if batch_size > 32:
        is_row_sharded = True
        assert batch_size % mesh_shape[0] == 0, "Batch size must be divisible by mesh rows"
        batch_size_per_device = batch_size // mesh_shape[0]
    else:
        is_row_sharded = False
        batch_size_per_device = batch_size

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Create router indices and weights
    router_indices = torch.zeros(batch_size * seq_len, num_experts_per_tok, dtype=torch.long)
    routing_weights = torch.zeros(batch_size * seq_len, num_experts)

    for b, s in itertools.product(range(batch_size), range(seq_len)):
        active_experts = torch.randperm(num_experts)[:num_experts_per_tok]
        router_indices[b * seq_len + s, :] = active_experts
        weights = torch.rand(num_experts_per_tok)
        weights = weights / weights.sum()
        routing_weights[b * seq_len + s, active_experts] = weights

    # Create dense topk weights for TTNN
    topk_weights_dense = torch.tensor(
        [[routing_weights[i, j].item() for j in b] for i, b in enumerate(router_indices)],
        dtype=torch.bfloat16,
    )

    # Create reference experts and get weights (same weights for both models)
    reference_experts, state_dict = _create_reference_experts_and_weights(config, num_experts)
    ref_output = gpt_oss_experts_reference(hidden_states, router_indices, routing_weights, reference_experts)

    # Create TTNN configs
    num_devices = mesh_device.get_num_devices()
    throughput_config = ThroughputExpertConfig(
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=num_devices,
    )

    # Use DRAM_MEMORY_CONFIG for decode to avoid L1 OOM with larger models (e.g., 120b with 4 experts/device)
    # This matches the configuration used in mlp.py
    dispatch_config = AllToAllDispatchConfig(cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    combine_config = AllToAllCombineConfig(cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    program_config = ThroughputProgramConfig()

    # Create expert mapping tensors
    expert_mapping_tensors = create_expert_mapping_tensors(
        num_devices=num_devices,
        num_experts_per_device=throughput_config.num_experts_per_device,
        mesh_device=mesh_device,
    )

    # Create remap topk mask (rows is dispatch dimension)
    num_dispatch_device_rows = mesh_shape[0]  # mesh_shape is defined earlier in this function
    remap_topk_mask = create_remap_topk_mask(
        num_dispatch_device_rows=num_dispatch_device_rows,
        num_experts=num_experts,
        mesh_device=mesh_device,
    )

    # Load expert weights (same weights as reference model)
    weights = load_throughput_expert_weights(
        mesh_device=mesh_device,
        config=throughput_config,
        state_dict=state_dict,
        weight_dtype=ttnn.bfloat16,
    )

    # Convert inputs to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )

    tt_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(1),  # [B, 1, S, H]
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )

    tt_routing_weights = ttnn.from_torch(
        topk_weights_dense.unsqueeze(1).unsqueeze(1),  # [B, 1, 1, K]
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )

    tt_router_indices = ttnn.from_torch(
        router_indices.unsqueeze(1).unsqueeze(1),  # [B, 1, 1, K]
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=mesh_mapper,
    )

    # Run TTNN implementation
    tt_output = gpt_oss_experts_ttnn(
        tt_hidden_states,
        tt_router_indices,
        tt_routing_weights,
        weights,
        throughput_config,
        expert_mapping_tensors,
        remap_topk_mask,
        dispatch_config,
        combine_config,
        program_config,
        mesh_device,
    )

    # Convert output to torch
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_shape))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., : batch_size * seq_len, :hidden_size]

    # Compare with reference
    passing, pcc = _compare_with_reference(
        tt_output_torch,
        ref_output,
        expected_pcc,
        expected_atol,
        expected_rtol,
        "experts",
    )
    assert passing, f"Experts test failed. PCC: {pcc} < {expected_pcc}"

    # Re-create input tensors for perf measurement since decode_forward deallocates them
    tt_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_routing_weights = ttnn.from_torch(
        topk_weights_dense.unsqueeze(1).unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_router_indices = ttnn.from_torch(
        router_indices.unsqueeze(1).unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=mesh_mapper,
    )

    # decode_forward deallocates hidden_states, topk_expert_indices, and
    # topk_expert_weights internally, so we must clone them each iteration.
    def op_fn():
        return gpt_oss_experts_ttnn(
            ttnn.clone(tt_hidden_states),
            ttnn.clone(tt_router_indices),
            ttnn.clone(tt_routing_weights),
            weights,
            throughput_config,
            expert_mapping_tensors,
            remap_topk_mask,
            dispatch_config,
            combine_config,
            program_config,
            mesh_device,
        )

    # Device performance measurement mode (when env var is set)
    if os.getenv(DEVICE_PERF_ENV_VAR) is not None:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        from tracy import signpost

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            output = op_fn()
            ttnn.deallocate(output)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                # NOTE: This sync inflates op-to-op latency but is required for
                # accurate per-iteration measurement. Production would pipeline
                # trace executions without per-iteration sync. See module docstring.
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                output = op_fn()
                # NOTE: This sync inflates op-to-op latency. See module docstring.
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
            trace_mode=trace_mode,
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
            target=expected_perf_us if expected_perf_us > 0 and not trace_mode and program_cache_enabled else None,
        )
        benchmark_data.save_partial_run_json(
            perf_profiler,
            run_type="gpt_oss_fused_ops",
            ml_model_name="gpt-oss",
            batch_size=batch_size,
            input_sequence_length=seq_len,
        )

        if expected_perf_us > 0 and not trace_mode and program_cache_enabled:
            perf_margin = 0.2
            assert perf_us <= expected_perf_us * (
                1 + perf_margin
            ), f"Perf regression: {perf_us:.3f}us exceeds expected {expected_perf_us:.3f}us"
        elif expected_perf_us == 0 and not trace_mode and program_cache_enabled:
            logger.warning("TODO: Set expected_perf_us using a measured baseline.")

    return pcc


def _skip_single_device_ccl():
    """Skip single device test because this fused op contains CCL ops."""
    pytest.skip(
        "Single-device test is not applicable because gpt_oss_experts includes CCL ops "
        "(all_to_all_dispatch, all_to_all_combine, all_reduce)."
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # Decode mode only - this fused op is for decode
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 0.998, 0.5, 0.5, 0.0),  # Measured PCC: 0.9983
    ],
)
@pytest.mark.parametrize("use_real_weights", [False], ids=["random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
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
def test_gpt_oss_experts(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Test the gpt_oss_experts fused op (full decode_forward).

    This tests the complete throughput experts forward pass including:
    - Tensor preparation
    - all_to_all_dispatch (CCL)
    - moe_expert_token_remap
    - Sparse matmul (gate/up/down) + SwiGLU
    - all_to_all_combine (CCL)
    - Routing weight application and reduction
    - all_reduce (CCL)
    """
    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

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

    # Batch size - matches decode_128 test configuration
    # In 4x8 mesh, batch 128 is distributed as 128/4 = 32 per row
    batch_size = 128

    # Run test
    pcc = _run_experts_test(
        mesh_device,
        config,
        batch_size,
        seq_len,
        hidden_size,
        num_experts,
        num_experts_per_tok,
        intermediate_size,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        use_real_weights,
        f"gpt_oss_experts_{mode}_seq{seq_len}",
    )

    logger.info(f"Test passed with PCC: {pcc}")


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.998, 0.5, 0.5, 0.0),
    ],
)
@pytest.mark.parametrize("use_real_weights", [False], ids=["random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
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
def test_gpt_oss_experts_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Single device test for gpt_oss_experts.

    This test is skipped because gpt_oss_experts contains CCL ops:
    - all_to_all_dispatch
    - all_to_all_combine
    - all_reduce
    """
    _skip_single_device_ccl()


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_gpt_oss_experts_device_perf(mode, seq_len):
    """Device performance test for gpt_oss_experts.

    This test measures device kernel duration and op-to-op latency using Tracy profiler.
    """
    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    # Batch size for decode_128 configuration
    batch_size = 128

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"gpt_oss_experts_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_experts.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_gpt_oss_experts -k "{expr}"'

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
