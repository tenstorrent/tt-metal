# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for gpt_oss_prepare_expert_tensors.

This fused op prepares tensors for all_to_all_dispatch in the throughput experts decode path.
It performs:
1. Reshape hidden_states from [B*S, 1, 1, H] to [B, 1, S, H]
2. Typecast and reshape expert indices from [B*S, 1, 1, K] to [B, 1, S, K] as uint16
3. Reshape expert weights from [B*S, 1, 1, K] to [B, 1, S, K]
4. Convert all tensors to ROW_MAJOR layout for all_to_all_dispatch

This is a decode-only fused op (seq_len=1).
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
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

DEVICE_PERF_ENV_VAR = "GPT_OSS_PREPARE_EXPERT_TENSORS_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
# TODO: Set device perf targets based on measured baselines
DEVICE_PERF_TARGETS_US = {}


def gpt_oss_prepare_expert_tensors_reference(
    hidden_states: torch.Tensor,
    topk_expert_indices: torch.Tensor,
    topk_expert_weights: torch.Tensor,
    batch_size_per_device: int,
    seq_len: int,
    hidden_size: int,
    num_experts_per_tok: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for prepare_expert_tensors.

    This performs the tensor preparation steps before all_to_all_dispatch:
    1. Reshape hidden_states to [B, 1, S, H]
    2. Convert expert indices to uint16 and reshape to [B, 1, S, K]
    3. Reshape expert weights to [B, 1, S, K]

    Args:
        hidden_states: Input tensor, shape depends on input layout
        topk_expert_indices: Expert indices tensor
        topk_expert_weights: Expert weights tensor
        batch_size_per_device: Batch size per device
        seq_len: Sequence length (always 1 for decode)
        hidden_size: Hidden dimension
        num_experts_per_tok: Number of experts per token (K)

    Returns:
        Tuple of (hidden_rm, topk_indices_rm, topk_expert_weights_reshaped)
    """
    # Reshape hidden_states: flatten to [-1, 1, 1, H] then reshape to [B, 1, S, H]
    hidden_flat = hidden_states.reshape(-1, 1, 1, hidden_size)
    hidden_rm = hidden_flat.reshape(batch_size_per_device, 1, seq_len, hidden_size)

    # Process expert indices: cast to int32, reshape, cast to int16, then final reshape
    # In PyTorch we'll use int32 then convert - matching the typecast sequence
    indices_uint32 = topk_expert_indices.to(torch.int32)
    indices_flat = indices_uint32.reshape(-1, 1, 1, num_experts_per_tok)
    indices_uint16 = indices_flat.to(torch.int16)
    topk_indices_rm = indices_uint16.reshape(batch_size_per_device, 1, seq_len, num_experts_per_tok)

    # Reshape expert weights
    weights_flat = topk_expert_weights.reshape(-1, 1, 1, num_experts_per_tok)
    topk_weights_reshaped = weights_flat.reshape(batch_size_per_device, 1, seq_len, num_experts_per_tok)

    return hidden_rm, topk_indices_rm, topk_weights_reshaped


def gpt_oss_prepare_expert_tensors_ttnn(
    hidden_states: ttnn.Tensor,
    topk_expert_indices: ttnn.Tensor,
    topk_expert_weights: ttnn.Tensor,
    batch_size_per_device: int,
    seq_len: int,
    hidden_size: int,
    num_experts_per_tok: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """TTNN implementation for prepare_expert_tensors.

    This is the exact sequence of ops from decode_forward that prepares
    tensors for all_to_all_dispatch.

    Args:
        hidden_states: Input tensor [B*S, 1, 1, H] or similar
        topk_expert_indices: Expert indices tensor
        topk_expert_weights: Expert weights tensor
        batch_size_per_device: Batch size per device
        seq_len: Sequence length (always 1 for decode)
        hidden_size: Hidden dimension
        num_experts_per_tok: Number of experts per token (K)

    Returns:
        Tuple of (hidden_rm, topk_indices_rm, topk_expert_weights unchanged)
    """
    # Step 1: Reshape hidden_states
    hidden_states = ttnn.reshape(hidden_states, (-1, 1, 1, hidden_size))

    # Step 2: Process expert indices
    topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint32)
    topk_expert_indices = ttnn.reshape(topk_expert_indices, (-1, 1, 1, num_experts_per_tok))
    topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint16)

    # Step 3: Reshape expert weights
    topk_expert_weights = ttnn.reshape(topk_expert_weights, (-1, 1, 1, num_experts_per_tok))

    # Step 4: Convert to ROW_MAJOR layout for all_to_all_dispatch
    hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_rm = ttnn.reshape(hidden_rm, shape=(batch_size_per_device, 1, seq_len, hidden_size))

    topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    topk_indices_rm = ttnn.reshape(topk_indices_rm, shape=(batch_size_per_device, 1, seq_len, num_experts_per_tok))

    # Note: topk_expert_weights is not converted to ROW_MAJOR here as it's used later
    # in the decode_forward after combine. We return it reshaped but in original layout.

    return hidden_rm, topk_indices_rm, topk_expert_weights


def _compare_with_reference(
    tt_output: torch.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    atol: float,
    rtol: float,
    name: str = "",
) -> None:
    """Compare TT output with reference, asserting PCC and tolerance."""
    passing, pcc = comp_pcc(ref_output.float(), tt_output.float(), expected_pcc)
    logger.info(f"PCC {name}: {pcc}")
    assert passing, f"PCC {name}: {pcc} is below required {expected_pcc}"
    # For integer tensors, skip the close check
    if ref_output.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        torch.testing.assert_close(tt_output.float(), ref_output.float(), rtol=rtol, atol=atol)


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
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            if output is not None:
                ttnn.deallocate(output)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                outputs = op_fn()
                for output in outputs:
                    if output is not None:
                        ttnn.deallocate(output)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            outputs = op_fn()
            for output in outputs:
                if output is not None:
                    ttnn.deallocate(output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("gpt_oss_prepare_expert_tensors_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("gpt_oss_prepare_expert_tensors_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("gpt_oss_prepare_expert_tensors_perf") * 1e6

    # Non-trace mode
    for _ in range(warmup_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            if output is not None:
                ttnn.deallocate(output)

    profiler.clear()
    profiler.start("gpt_oss_prepare_expert_tensors_perf")
    for _ in range(measure_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            if output is not None:
                ttnn.deallocate(output)
    profiler.end("gpt_oss_prepare_expert_tensors_perf", PERF_CNT=measure_iters)
    return profiler.get("gpt_oss_prepare_expert_tensors_perf") * 1e6


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


def _run_prepare_expert_tensors_test(
    mesh_device: ttnn.MeshDevice,
    tt_hidden_states: ttnn.Tensor,
    tt_topk_indices: ttnn.Tensor,
    tt_topk_weights: ttnn.Tensor,
    ref_hidden_rm: torch.Tensor,
    ref_indices_rm: torch.Tensor,
    ref_weights: torch.Tensor,
    batch_size_per_device: int,
    seq_len: int,
    hidden_size: int,
    num_experts_per_tok: int,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    step_prefix: str,
):
    """Run the prepare_expert_tensors test with accuracy and performance checks."""

    # Run the TTNN implementation
    tt_hidden_rm, tt_indices_rm, tt_weights = gpt_oss_prepare_expert_tensors_ttnn(
        tt_hidden_states,
        tt_topk_indices,
        tt_topk_weights,
        batch_size_per_device,
        seq_len,
        hidden_size,
        num_experts_per_tok,
    )

    # Convert to torch for comparison
    tt_hidden_rm_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden_rm)[0])
    tt_indices_rm_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_indices_rm)[0])
    tt_weights_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_weights)[0])

    # Compare outputs - hidden_states
    _compare_with_reference(
        tt_hidden_rm_torch,
        ref_hidden_rm,
        expected_pcc,
        expected_atol,
        expected_rtol,
        "hidden_rm",
    )

    # Compare outputs - indices (use exact match for integers)
    indices_match = torch.all(tt_indices_rm_torch == ref_indices_rm)
    logger.info(f"Indices exact match: {indices_match}")
    assert indices_match, "Expert indices don't match exactly"

    # Compare outputs - weights
    _compare_with_reference(
        tt_weights_torch.reshape(ref_weights.shape),
        ref_weights,
        expected_pcc,
        expected_atol,
        expected_rtol,
        "weights",
    )

    def op_fn():
        return gpt_oss_prepare_expert_tensors_ttnn(
            tt_hidden_states,
            tt_topk_indices,
            tt_topk_weights,
            batch_size_per_device,
            seq_len,
            hidden_size,
            num_experts_per_tok,
        )

    # Device performance measurement mode (when env var is set)
    if os.getenv(DEVICE_PERF_ENV_VAR) is not None:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        from tracy import signpost

        for _ in range(PERF_WARMUP_ITERS):
            outputs = op_fn()
            ttnn.synchronize_device(mesh_device)
            for output in outputs:
                if output is not None:
                    ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outputs = op_fn()
            for output in outputs:
                if output is not None:
                    ttnn.deallocate(output)
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
                outputs = op_fn()
                ttnn.synchronize_device(mesh_device)
                for output in outputs:
                    if output is not None:
                        ttnn.deallocate(output)
            signpost("stop")
        return

    # Standard e2e performance measurement
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
        batch_size=batch_size_per_device,
        input_sequence_length=seq_len,
    )

    if expected_perf_us > 0 and not trace_mode and program_cache_enabled:
        perf_margin = 0.2
        assert perf_us <= expected_perf_us * (
            1 + perf_margin
        ), f"Perf regression: {perf_us:.3f}us exceeds expected {expected_perf_us:.3f}us"
    elif expected_perf_us == 0 and not trace_mode and program_cache_enabled:
        logger.warning("TODO: Set expected_perf_us using a measured baseline.")


def _build_inputs(
    mesh_device: ttnn.MeshDevice,
    batch_size_per_device: int,
    seq_len: int,
    hidden_size: int,
    num_experts_per_tok: int,
    num_experts: int,
    use_real_weights: bool,
):
    """Build input tensors for the test.

    Args:
        mesh_device: TTNN mesh device
        batch_size_per_device: Batch size per device (local batch)
        seq_len: Sequence length (1 for decode)
        hidden_size: Hidden dimension
        num_experts_per_tok: Number of experts per token (K)
        num_experts: Total number of experts
        use_real_weights: Whether to use real weights (not applicable for this op)

    Returns:
        Tuple of (tt_hidden, tt_indices, tt_weights, ref_hidden_rm, ref_indices_rm, ref_weights)
    """
    # Create random input tensors with shape as they come into decode_forward
    # hidden_states: [batch_size_per_device, 1, 1, hidden_size] - from reshape before this sequence
    torch_hidden = torch.randn(batch_size_per_device, 1, seq_len, hidden_size, dtype=torch.bfloat16)

    # topk_expert_indices: [batch_size_per_device, 1, 1, num_experts_per_tok] - from router
    torch_indices = torch.randint(0, num_experts, (batch_size_per_device, 1, seq_len, num_experts_per_tok))

    # topk_expert_weights: [batch_size_per_device, 1, 1, num_experts_per_tok] - from router
    torch_weights = torch.rand(batch_size_per_device, 1, seq_len, num_experts_per_tok, dtype=torch.bfloat16)
    # Normalize weights per token
    torch_weights = torch_weights / torch_weights.sum(dim=-1, keepdim=True)

    # Compute reference outputs
    ref_hidden_rm, ref_indices_rm, ref_weights = gpt_oss_prepare_expert_tensors_reference(
        torch_hidden,
        torch_indices,
        torch_weights,
        batch_size_per_device,
        seq_len,
        hidden_size,
        num_experts_per_tok,
    )

    # Create TTNN tensors - replicate across mesh for this test
    # The actual module uses ShardTensor2dMesh but for this unit test we replicate
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_hidden = ttnn.from_torch(
        torch_hidden,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_indices = ttnn.from_torch(
        torch_indices,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        dtype=ttnn.uint16,  # Start as uint16, will be typecast in the op
        layout=ttnn.TILE_LAYOUT,
    )

    tt_weights = ttnn.from_torch(
        torch_weights,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    return tt_hidden, tt_indices, tt_weights, ref_hidden_rm, ref_indices_rm, ref_weights


def _skip_single_device_no_ccl():
    """Skip message for single device test when op doesn't have CCL."""
    # This op doesn't have CCL ops, so single device test is applicable


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # Decode mode only - this is a decode-only fused op
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 1.0, 0.1, 0.1, 464.489),  # Measured baseline, PCC=1.0 for reshape/typecast ops
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
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
def test_gpt_oss_prepare_expert_tensors(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Test the gpt_oss_prepare_expert_tensors fused op.

    This tests the tensor preparation sequence before all_to_all_dispatch
    in the throughput experts decode path.
    """
    # Configuration from gpt-oss 20b model
    hidden_size = 2880
    num_experts = 128
    num_experts_per_tok = 4

    # Batch size per device - matches decode_128 test configuration
    # In 4x8 mesh, batch 128 is distributed as 128/4 = 32 per row
    batch_size_per_device = 32

    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # Build inputs
    tt_hidden, tt_indices, tt_weights, ref_hidden_rm, ref_indices_rm, ref_weights = _build_inputs(
        mesh_device,
        batch_size_per_device,
        seq_len,
        hidden_size,
        num_experts_per_tok,
        num_experts,
        use_real_weights=False,
    )

    # Run test
    _run_prepare_expert_tensors_test(
        mesh_device,
        tt_hidden,
        tt_indices,
        tt_weights,
        ref_hidden_rm,
        ref_indices_rm,
        ref_weights,
        batch_size_per_device,
        seq_len,
        hidden_size,
        num_experts_per_tok,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        f"gpt_oss_prepare_expert_tensors_{mode}_seq{seq_len}",
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 1.0, 0.1, 0.1, 464.489),  # Same as multi-device since no CCL
    ],
)
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
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
def test_gpt_oss_prepare_expert_tensors_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Single device test for gpt_oss_prepare_expert_tensors.

    This op doesn't contain CCL ops, so we can run it on a single device.
    The input shape for single device is the per-device chunk from multi-device.
    """
    # Configuration from gpt-oss 20b model
    hidden_size = 2880
    num_experts = 128
    num_experts_per_tok = 4

    # For single device test, we use the per-device batch size
    batch_size_per_device = 32

    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # Create a 1x1 submesh to get a single device
    single_device_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 1)))

    # Build inputs for single device
    torch_hidden = torch.randn(batch_size_per_device, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    torch_indices = torch.randint(0, num_experts, (batch_size_per_device, 1, seq_len, num_experts_per_tok))
    torch_weights = torch.rand(batch_size_per_device, 1, seq_len, num_experts_per_tok, dtype=torch.bfloat16)
    torch_weights = torch_weights / torch_weights.sum(dim=-1, keepdim=True)

    # Reference outputs
    ref_hidden_rm, ref_indices_rm, ref_weights = gpt_oss_prepare_expert_tensors_reference(
        torch_hidden,
        torch_indices,
        torch_weights,
        batch_size_per_device,
        seq_len,
        hidden_size,
        num_experts_per_tok,
    )

    # Create tensors on single device submesh
    tt_hidden = ttnn.from_torch(
        torch_hidden,
        device=single_device_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_indices = ttnn.from_torch(
        torch_indices,
        device=single_device_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_weights = ttnn.from_torch(
        torch_weights,
        device=single_device_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN implementation
    tt_hidden_rm, tt_indices_rm, tt_weights_out = gpt_oss_prepare_expert_tensors_ttnn(
        tt_hidden,
        tt_indices,
        tt_weights,
        batch_size_per_device,
        seq_len,
        hidden_size,
        num_experts_per_tok,
    )

    # Convert to torch - get from first device
    tt_hidden_rm_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden_rm)[0])
    tt_indices_rm_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_indices_rm)[0])
    tt_weights_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_weights_out)[0])

    # Compare outputs
    _compare_with_reference(
        tt_hidden_rm_torch,
        ref_hidden_rm,
        expected_pcc,
        expected_atol,
        expected_rtol,
        "hidden_rm (single device)",
    )

    indices_match = torch.all(tt_indices_rm_torch == ref_indices_rm)
    logger.info(f"Indices exact match (single device): {indices_match}")
    assert indices_match, "Expert indices don't match exactly (single device)"

    _compare_with_reference(
        tt_weights_torch.reshape(ref_weights.shape),
        ref_weights,
        expected_pcc,
        expected_atol,
        expected_rtol,
        "weights (single device)",
    )

    logger.info("Single device test passed!")


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_gpt_oss_prepare_expert_tensors_device_perf(mode, seq_len):
    """Device performance test for gpt_oss_prepare_expert_tensors.

    This test measures device kernel duration and op-to-op latency using Tracy profiler.
    """
    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    # Batch size per device for decode_128 configuration
    batch_size_per_device = 32

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"gpt_oss_prepare_expert_tensors_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_prepare_expert_tensors.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_gpt_oss_prepare_expert_tensors -k "{expr}"'

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
        batch_size=batch_size_per_device,
        input_sequence_length=seq_len,
    )


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_gpt_oss_prepare_expert_tensors_single_device_perf(mode, seq_len):
    """Single-device performance test for gpt_oss_prepare_expert_tensors.

    This test measures device kernel duration and op-to-op latency using Tracy profiler
    on a single device. Since this op doesn't have CCL ops, it can run on single device.
    """
    assert mode == "decode", "This is a decode-only fused op"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    # Batch size per device for decode_128 configuration
    batch_size_per_device = 32

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"gpt_oss_prepare_expert_tensors_single_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_prepare_expert_tensors.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_gpt_oss_prepare_expert_tensors_single_device -k "{expr}"'

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
        run_type="gpt_oss_fused_ops_single_device_perf",
        ml_model_name="gpt-oss",
        batch_size=batch_size_per_device,
        input_sequence_length=seq_len,
    )
