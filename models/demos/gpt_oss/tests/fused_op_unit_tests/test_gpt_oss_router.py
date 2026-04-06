# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for gpt_oss_router (TopKRouter.__call__).

This fused op tests the entire TopKRouter forward pass including:
1. ttnn.reshape - Reshape input from [B, S, H] to [B*S, H]
2. ttnn.linear - Apply router projection with weight and bias
3. ttnn.to_memory_config - Convert to DRAM (decode mode)
4. ttnn.typecast - Convert to bfloat16 (if needed)
5. ttnn.topk - Select top-k experts
6. ttnn.softmax - Normalize expert weights with HiFi4 fidelity
7. (Optional) ttnn.scatter - Scatter weights to full dimension (non-throughput only)

This is a decode-only fused op (seq_len=1).
No CCL ops, so single device test is supported.
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
from models.demos.gpt_oss.tt.topk import TopKRouter
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

DEVICE_PERF_ENV_VAR = "GPT_OSS_ROUTER_DEVICE_PERF"
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


def gpt_oss_router_reference(
    hidden_states: torch.Tensor,
    reference_router,
    use_throughput_experts: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for gpt_oss_router.

    Uses the HuggingFace GptOss router module as the reference.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        reference_router: HuggingFace GptOss router module (reference_layer.mlp.router)
        use_throughput_experts: Whether to return dense weights (throughput) or sparse (standard)

    Returns:
        Tuple of:
            - router_indices: Selected expert indices [batch * seq_len, num_experts_per_tok]
            - router_weights: Routing weights (dense or sparse depending on use_throughput_experts)
    """
    with torch.no_grad():
        # Reference router returns (routing_weights, routing_indices)
        router_scores, router_indices = reference_router(hidden_states)

    if use_throughput_experts:
        # When using throughput experts, convert sparse router_scores to dense router_weights
        # (reorder weights to match the order of the indices)
        dense_router_scores = torch.concat(
            [
                torch.tensor(
                    [router_scores[user, router_indices[user, i]] for i in range(router_indices.shape[1])]
                ).reshape(1, -1)
                for user in range(router_scores.shape[0])
            ],
            dim=0,
        )
        router_scores = dense_router_scores

    return router_indices, router_scores


def gpt_oss_router_ttnn(
    hidden_states: ttnn.Tensor,
    tt_router: TopKRouter,
    use_throughput_experts: bool,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """TTNN implementation for gpt_oss_router.

    This is the full TopKRouter.__call__ method.

    Args:
        hidden_states: Input tensor [batch, 1, seq_len, hidden_size] in TTNN format
        tt_router: TTNN TopKRouter module
        use_throughput_experts: Whether to return dense weights (throughput) or sparse (standard)

    Returns:
        Tuple of:
            - expert_indices: Selected expert indices
            - expert_weights: Routing weights (dense or sparse)
    """
    return tt_router(hidden_states, use_throughput_experts)


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
        indices, weights = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(indices)
        ttnn.deallocate(weights)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                indices, weights = op_fn()
                ttnn.deallocate(indices)
                ttnn.deallocate(weights)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            indices, weights = op_fn()
            ttnn.deallocate(indices)
            ttnn.deallocate(weights)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("gpt_oss_router_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("gpt_oss_router_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("gpt_oss_router_perf") * 1e6

    # Non-trace mode
    for _ in range(warmup_iters):
        indices, weights = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(indices)
        ttnn.deallocate(weights)

    profiler.clear()
    profiler.start("gpt_oss_router_perf")
    for _ in range(measure_iters):
        indices, weights = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(indices)
        ttnn.deallocate(weights)
    profiler.end("gpt_oss_router_perf", PERF_CNT=measure_iters)
    return profiler.get("gpt_oss_router_perf") * 1e6


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

    # Calculate total kernel/op-to-op time across ALL ops
    total_kernel_ns = sum(entry["total_kernel_duration_ns"] for entry in op_stats.values())
    total_op_to_op_ns = sum(entry["total_op_to_op_latency_ns"] for entry in op_stats.values())
    return op_stats, total_kernel_ns, total_op_to_op_ns


def _create_reference_router_and_state_dict(config, layer_idx=0):
    """Create HuggingFace reference router and extract state dict.

    Returns both the reference router module and a state dict compatible with
    TopKRouter initialization.
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    reference_layer = GptOssDecoderLayer(config, layer_idx=layer_idx)
    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, 1)

    reference_router = reference_layer.mlp.router

    # Extract router weights from reference model
    # HuggingFace format: weight [num_experts, hidden_size], bias [num_experts]
    state_dict = {
        "weight": reference_router.weight.data.clone(),
        "bias": reference_router.bias.data.clone(),
    }

    return reference_router, state_dict


def _run_router_test(
    mesh_device: ttnn.MeshDevice,
    config,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    use_real_weights: bool,
    step_prefix: str,
):
    """Run the full router fused op test."""

    # Determine batch per device and row sharding
    mesh_shape = mesh_device.shape
    if batch_size > 32:
        is_row_sharded = True
        assert batch_size % mesh_shape[0] == 0, "Batch size must be divisible by mesh rows"
        batch_size_per_device = batch_size // mesh_shape[0]
    else:
        is_row_sharded = False
        batch_size_per_device = batch_size

    # Use throughput experts mode when mesh has multiple rows (like 4x8)
    use_throughput_experts = mesh_shape[0] > 1 and batch_size * seq_len > 1

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Create reference router and get weights (same weights for both models)
    reference_router, state_dict = _create_reference_router_and_state_dict(config)
    ref_indices, ref_weights = gpt_oss_router_reference(hidden_states, reference_router, use_throughput_experts)

    # Create TTNN router
    tt_router = TopKRouter(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        tensor_cache_path=None,
    )

    # Convert inputs to TTNN tensors
    # Router expects input shape: [batch, seq_len, hidden_dim] -> reshaped to [batch*seq_len, hidden_dim]
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )

    # Match the module test input shape: [batch*seq_len, 1, hidden_size]
    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(-1, 1, hidden_size),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=mesh_mapper,
    )

    # Run TTNN implementation
    tt_indices, tt_weights = gpt_oss_router_ttnn(tt_hidden_states, tt_router, use_throughput_experts)

    # Convert output to torch
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_shape))
    tt_indices_torch = ttnn.to_torch(tt_indices, mesh_composer=mesh_composer)[:batch_size, :num_experts_per_tok]
    tt_weights_torch = ttnn.to_torch(tt_weights, mesh_composer=mesh_composer)[:batch_size, :num_experts_per_tok]

    # Compare with reference
    # Sort indices for comparison since order may differ between implementations
    sorted_tt_indices, sorted_tt_indices_order = torch.sort(tt_indices_torch, dim=-1)
    sorted_ref_indices, sorted_ref_indices_order = torch.sort(ref_indices, dim=-1)

    indices_passing, indices_pcc = _compare_with_reference(
        sorted_tt_indices,
        sorted_ref_indices,
        expected_pcc,
        expected_atol,
        expected_rtol,
        "router_indices",
    )

    # Reorder weights to match sorted indices
    weights_passing, weights_pcc = _compare_with_reference(
        tt_weights_torch.squeeze()[sorted_tt_indices_order],
        ref_weights.squeeze()[sorted_ref_indices_order],
        expected_pcc,
        expected_atol,
        expected_rtol,
        "router_weights",
    )

    assert indices_passing, f"Router indices test failed. PCC: {indices_pcc} < {expected_pcc}"
    assert weights_passing, f"Router weights test failed. PCC: {weights_pcc} < {expected_pcc}"

    def op_fn():
        return gpt_oss_router_ttnn(tt_hidden_states, tt_router, use_throughput_experts)

    # Device performance measurement mode (when env var is set)
    if os.getenv(DEVICE_PERF_ENV_VAR) is not None:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        from tracy import signpost

        for _ in range(PERF_WARMUP_ITERS):
            indices, weights = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(indices)
            ttnn.deallocate(weights)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            indices, weights = op_fn()
            ttnn.deallocate(indices)
            ttnn.deallocate(weights)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                indices, weights = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(indices)
                ttnn.deallocate(weights)
            signpost("stop")
        return indices_pcc, weights_pcc

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

    return indices_pcc, weights_pcc


def _run_router_single_device_test(
    single_device_mesh: ttnn.MeshDevice,
    config,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    use_real_weights: bool,
    step_prefix: str,
):
    """Run the router fused op test on a single device (1x1 submesh).

    The input shape is the per-device chunk from the multi-device test.
    For decode_128 on 4x8 mesh, per-device batch is 128/4 = 32.
    """

    # Single device always uses non-throughput experts (no CCL)
    use_throughput_experts = False

    # Create input tensors (per-device shape)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Create reference router and get weights
    reference_router, state_dict = _create_reference_router_and_state_dict(config)
    ref_indices, ref_weights = gpt_oss_router_reference(hidden_states, reference_router, use_throughput_experts)

    # Create TTNN router on single device mesh
    tt_router = TopKRouter(
        mesh_device=single_device_mesh,
        hf_config=config,
        state_dict=state_dict,
        tensor_cache_path=None,
    )

    # Convert inputs to TTNN tensors (no mesh mapper for single device)
    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(-1, 1, hidden_size),
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # Run TTNN implementation
    tt_indices, tt_weights = gpt_oss_router_ttnn(tt_hidden_states, tt_router, use_throughput_experts)

    # Convert output to torch - use get_device_tensors to get tensor from first device
    tt_indices_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_indices)[0])[:batch_size, :num_experts_per_tok]
    tt_weights_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_weights)[0])[
        :batch_size, :num_experts
    ]  # Non-throughput has sparse weights

    # Compare with reference - for non-throughput, weights are sparse [batch, num_experts]
    sorted_tt_indices, sorted_tt_indices_order = torch.sort(tt_indices_torch, dim=-1)
    sorted_ref_indices, sorted_ref_indices_order = torch.sort(ref_indices, dim=-1)

    indices_passing, indices_pcc = _compare_with_reference(
        sorted_tt_indices,
        sorted_ref_indices,
        expected_pcc,
        expected_atol,
        expected_rtol,
        "router_indices_single_device",
    )

    # For non-throughput experts, ref_weights is sparse [batch, num_experts]
    # We need to compare the weights at the selected expert indices
    tt_weights_at_indices = torch.gather(tt_weights_torch, dim=1, index=tt_indices_torch.long())
    ref_weights_at_indices = torch.gather(ref_weights, dim=1, index=ref_indices.long())

    weights_passing, weights_pcc = _compare_with_reference(
        tt_weights_at_indices.squeeze()[sorted_tt_indices_order],
        ref_weights_at_indices.squeeze()[sorted_ref_indices_order],
        expected_pcc,
        expected_atol,
        expected_rtol,
        "router_weights_single_device",
    )

    assert indices_passing, f"Router indices single device test failed. PCC: {indices_pcc} < {expected_pcc}"
    assert weights_passing, f"Router weights single device test failed. PCC: {weights_pcc} < {expected_pcc}"

    def op_fn():
        return gpt_oss_router_ttnn(tt_hidden_states, tt_router, use_throughput_experts)

    # Performance measurement
    if not trace_mode or program_cache_enabled:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_single_device_{trace_suffix}_{cache_suffix}"

        perf_profiler.start("run")
        perf_profiler.start(step_name)
        perf_us = _measure_perf_us(
            single_device_mesh,
            op_fn,
            PERF_WARMUP_ITERS,
            PERF_MEASURE_ITERS,
            trace_mode=trace_mode,
        )
        logger.info(f"Single device perf avg: {perf_us:.3f} us over {PERF_MEASURE_ITERS} iters")
        perf_profiler.end(step_name)
        perf_profiler.end("run")

        benchmark_data.add_measurement(
            perf_profiler,
            0,
            step_name,
            f"{step_name}-avg_us",
            perf_us,
            step_warm_up_num_iterations=PERF_WARMUP_ITERS,
        )
        benchmark_data.save_partial_run_json(
            perf_profiler,
            run_type="gpt_oss_fused_ops_single_device",
            ml_model_name="gpt-oss",
            batch_size=batch_size,
            input_sequence_length=seq_len,
        )

    # Clean up tensors on the device to allow proper teardown
    ttnn.deallocate(tt_hidden_states)
    ttnn.deallocate(tt_indices)
    ttnn.deallocate(tt_weights)
    ttnn.synchronize_device(single_device_mesh)

    return indices_pcc, weights_pcc


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # Decode mode only - this fused op is for decode
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        # PCC can vary with random weights due to bf8_b input precision:
        # indices PCC ~0.98-0.99, weights PCC ~0.92-1.0
        ("decode", 1, 0.90, 0.5, 0.5, 0.0),
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
def test_gpt_oss_router(
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
    """Test the gpt_oss_router fused op (TopKRouter.__call__).

    This tests the complete router forward pass including:
    - Reshape input
    - Linear projection (weight + bias)
    - Memory config conversion
    - Typecast (if needed)
    - TopK selection
    - Softmax normalization
    - (Optional) Scatter for non-throughput experts
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

    # Batch size - matches decode_128 test configuration
    # In 4x8 mesh, batch 128 is distributed as 128/4 = 32 per row
    batch_size = 128

    # Run test
    indices_pcc, weights_pcc = _run_router_test(
        mesh_device,
        config,
        batch_size,
        seq_len,
        hidden_size,
        num_experts,
        num_experts_per_tok,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        use_real_weights,
        f"gpt_oss_router_{mode}_seq{seq_len}",
    )

    logger.info(f"Test passed with indices PCC: {indices_pcc}, weights PCC: {weights_pcc}")


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.98, 0.5, 0.5, 0.0),  # Measured PCC: indices=0.98-0.99, weights=0.99
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
def test_gpt_oss_router_single_device(
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
    """Single device test for gpt_oss_router.

    Unlike experts, the router has no CCL ops so single device test is supported.
    Uses a 1x1 submesh from the full mesh and runs with per-device batch size.

    Note: There is a known infrastructure issue with submesh cleanup during pytest
    teardown that may cause a crash after the test passes. The test results are valid.
    This affects all single device tests that use mesh_device.create_submesh().
    See conftest.py submesh cleanup logic.
    """
    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    # Create a 1x1 submesh to get a single device
    single_device_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 1)))

    if not program_cache_enabled:
        single_device_mesh.disable_and_clear_program_cache()

    # Get HF config
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Use config values for dimensions
    hidden_size = config.hidden_size
    num_experts = config.num_local_experts
    num_experts_per_tok = config.num_experts_per_tok

    # Per-device batch size: 128 / 4 rows = 32
    batch_size_per_device = 32

    # Run single device test
    indices_pcc, weights_pcc = _run_router_single_device_test(
        single_device_mesh,
        config,
        batch_size_per_device,
        seq_len,
        hidden_size,
        num_experts,
        num_experts_per_tok,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        use_real_weights,
        f"gpt_oss_router_{mode}_seq{seq_len}",
    )

    logger.info(f"Single device test passed with indices PCC: {indices_pcc}, weights PCC: {weights_pcc}")


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_gpt_oss_router_device_perf(mode, seq_len):
    """Device performance test for gpt_oss_router.

    This test measures device kernel duration and op-to-op latency using Tracy profiler.
    """
    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    # Batch size for decode_128 configuration
    batch_size = 128

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"gpt_oss_router_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_router.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_gpt_oss_router -k "{expr}"'

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


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_gpt_oss_router_single_device_device_perf(mode, seq_len):
    """Single device device performance test for gpt_oss_router.

    This test runs on a single device to measure per-device kernel duration.
    """
    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    # Per-device batch size
    batch_size = 32

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"gpt_oss_router_single_device_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_router.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_gpt_oss_router_single_device -k "{expr}"'

    perf_profiler.start("run")
    perf_profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = _collect_device_perf(
        command,
        subdir="gpt_oss_fused_ops_single_device_device_perf",
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
    logger.info(f"Single device perf per-op averages (ns): {json.dumps(op_stats, indent=2)}")
    logger.info(f"Single device perf totals: kernel={total_kernel_us:.3f} us, op_to_op={total_op_to_op_us:.3f} us")
    logger.info(
        f"Single device perf per-iteration averages: kernel={avg_kernel_us:.3f} us, op_to_op={avg_op_to_op_us:.3f} us"
    )

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
        run_type="gpt_oss_fused_ops_single_device_device_perf",
        ml_model_name="gpt-oss",
        batch_size=batch_size,
        input_sequence_length=seq_len,
    )
