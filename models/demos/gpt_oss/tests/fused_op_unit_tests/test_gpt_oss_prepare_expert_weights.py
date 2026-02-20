# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for gpt_oss_prepare_expert_weights.

This fused op prepares routing weights for element-wise multiplication with expert outputs:
1. Reshape from [B, 1, S, K] to [-1, 1, 1, K]
2. Layout conversion to ROW_MAJOR
3. Repeat to expand hidden dimension
4. Permute to [K, 1, B*S, H]
5. Layout conversion to TILE
6. Deallocate intermediate tensor

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
from models.demos.gpt_oss.tt.experts_throughput.decode import prepare_expert_weights
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

# ==============================================================================
# Constants
# ==============================================================================
DEVICE_PERF_ENV_VAR = "GPT_OSS_PREPARE_EXPERT_WEIGHTS_DEVICE_PERF"
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
# PyTorch Reference Implementation
# ==============================================================================
def gpt_oss_prepare_expert_weights_reference(
    topk_expert_weights: torch.Tensor,
    num_experts_per_tok: int,
    hidden_size: int,
) -> torch.Tensor:
    """PyTorch reference implementation for prepare_expert_weights.

    Transforms routing weights from [1, 1, tokens, K] to [K, 1, tokens, 1] format.
    In production, broadcasting in the subsequent ttnn.mul expands the last dim to H.

    Args:
        topk_expert_weights: Routing weights [1, 1, tokens_per_device, num_experts_per_tok]
        num_experts_per_tok: Number of experts selected per token (K)
        hidden_size: Hidden dimension size (H) - unused, kept for API compat

    Returns:
        Transformed weights tensor [K, 1, tokens, 1]
    """
    # Permute: [1, 1, tokens, K] -> [K, 1, tokens, 1]
    weights = topk_expert_weights.permute(3, 1, 2, 0)
    return weights


# ==============================================================================
# TTNN Implementation
# ==============================================================================
def gpt_oss_prepare_expert_weights_ttnn(
    topk_expert_weights: ttnn.Tensor,
    num_experts_per_tok: int,
    hidden_size: int,
) -> ttnn.Tensor:
    """TTNN implementation for prepare_expert_weights.

    This is exactly the prepare_expert_weights function from decode.py.

    Args:
        topk_expert_weights: Routing weights [batch_size, 1, seq_len, num_experts_per_tok]
        num_experts_per_tok: Number of experts selected per token (K)
        hidden_size: Hidden dimension size (H)

    Returns:
        Transformed weights tensor [K, 1, B*S, H] in TILE layout
    """
    return prepare_expert_weights(
        topk_expert_weights=topk_expert_weights,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
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
        profiler.start("fused_op_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("fused_op_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("fused_op_perf") * 1e6

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


# ==============================================================================
# Test Implementation
# ==============================================================================
def _run_prepare_expert_weights_test(
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    seq_len: int,
    num_experts_per_tok: int,
    hidden_size: int,
    expected_pcc: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    step_prefix: str,
):
    """Run the prepare_expert_weights fused op test."""
    mesh_shape = mesh_device.shape

    # Determine tokens per device (matching production shape from decode_forward)
    if batch_size > 32:
        assert batch_size % mesh_shape[0] == 0, "Batch size must be divisible by mesh rows"
        batch_size_per_device = batch_size // mesh_shape[0]
    else:
        batch_size_per_device = batch_size
    tokens_per_device = batch_size_per_device * seq_len

    # Create input tensor matching production shape: [1, 1, tokens_per_device, K]
    # In decode_forward, topk_expert_weights is reshaped to this before calling prepare_expert_weights
    topk_weights_torch = torch.randn(1, 1, tokens_per_device, num_experts_per_tok, dtype=torch.bfloat16)
    # Normalize weights (as they would be in real model)
    topk_weights_torch = torch.softmax(topk_weights_torch.float(), dim=-1).to(torch.bfloat16)

    # Reference computation
    ref_output = gpt_oss_prepare_expert_weights_reference(
        topk_weights_torch,
        num_experts_per_tok,
        hidden_size,
    )

    # Convert to TTNN tensor (replicate across mesh - each device processes its own tokens independently)
    tt_topk_weights = ttnn.from_torch(
        topk_weights_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run TTNN implementation
    tt_output = gpt_oss_prepare_expert_weights_ttnn(
        tt_topk_weights,
        num_experts_per_tok,
        hidden_size,
    )

    # Convert output to torch (get first device since all are identical)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    # Slice to expected dimensions (remove tile padding)
    tt_output_torch = tt_output_torch[:num_experts_per_tok, :1, :tokens_per_device, :1]

    # Compare with reference
    passing, pcc = _compare_with_reference(
        tt_output_torch,
        ref_output,
        expected_pcc,
        "prepare_expert_weights",
    )
    assert passing, f"prepare_expert_weights test failed. PCC: {pcc} < {expected_pcc}"

    # Re-create input tensor for perf measurement since prepare_expert_weights deallocates it
    tt_topk_weights = ttnn.from_torch(
        topk_weights_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # prepare_expert_weights deallocates its input internally, so we must
    # clone the input tensor for each iteration to avoid "Buffer not allocated" errors.
    def op_fn():
        return gpt_oss_prepare_expert_weights_ttnn(
            ttnn.clone(tt_topk_weights),
            num_experts_per_tok,
            hidden_size,
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


# ==============================================================================
# Pytest Test Functions
# ==============================================================================
@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_perf_us",
    [
        # Decode mode (seq_len=1) - Measured PCC: 1.0
        ("decode", 1, 0.999, 0.0),  # TODO: Set expected_perf_us based on measured baseline
        # Prefill modes
        ("prefill", 128, 0.999, 0.0),
        ("prefill", 1024, 0.999, 0.0),
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
def test_gpt_oss_prepare_expert_weights(
    mode,
    seq_len,
    expected_pcc,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Test the gpt_oss_prepare_expert_weights fused op.

    This tests the routing weight preparation:
    - Reshape from [B, 1, S, K] to [-1, 1, 1, K]
    - Layout conversion to ROW_MAJOR
    - Repeat to expand hidden dimension
    - Permute to [K, 1, B*S, H]
    - Layout conversion to TILE
    """
    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # Get HF config
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Use config values for dimensions
    hidden_size = config.hidden_size
    num_experts_per_tok = config.num_experts_per_tok

    # Batch size based on mode
    batch_size = 128 if mode == "decode" else 1

    pcc = _run_prepare_expert_weights_test(
        mesh_device,
        batch_size,
        seq_len,
        num_experts_per_tok,
        hidden_size,
        expected_pcc,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        f"gpt_oss_prepare_expert_weights_{mode}_seq{seq_len}",
    )

    logger.info(f"Test passed with PCC: {pcc}")


# ==============================================================================
# Single Device Test
# ==============================================================================
@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_perf_us",
    [
        ("decode", 1, 0.999, 0.0),  # Measured PCC: 1.0
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
def test_gpt_oss_prepare_expert_weights_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Single device test for gpt_oss_prepare_expert_weights.

    This test can run on single device since prepare_expert_weights
    does not contain CCL ops.
    """
    # Create a 1x1 submesh to get a single device
    single_device_mesh = mesh_device.create_submesh(ttnn.MeshShape((1, 1)))

    # Get HF config
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    hidden_size = config.hidden_size
    num_experts_per_tok = config.num_experts_per_tok
    batch_size_per_device = 32  # Single device batch size (128 / 4 rows)
    tokens_per_device = batch_size_per_device * seq_len

    # Create input tensor matching production shape: [1, 1, tokens_per_device, K]
    topk_weights_torch = torch.randn(1, 1, tokens_per_device, num_experts_per_tok, dtype=torch.bfloat16)
    topk_weights_torch = torch.softmax(topk_weights_torch.float(), dim=-1).to(torch.bfloat16)

    # Reference computation
    ref_output = gpt_oss_prepare_expert_weights_reference(
        topk_weights_torch,
        num_experts_per_tok,
        hidden_size,
    )

    # Convert to TTNN tensor (single device submesh)
    tt_topk_weights = ttnn.from_torch(
        topk_weights_torch,
        device=single_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(single_device_mesh),
    )

    # Run TTNN implementation
    tt_output = gpt_oss_prepare_expert_weights_ttnn(
        tt_topk_weights,
        num_experts_per_tok,
        hidden_size,
    )

    # Convert output to torch
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_output_torch = tt_output_torch[:num_experts_per_tok, :1, :tokens_per_device, :1]

    # Compare with reference
    passing, pcc = _compare_with_reference(
        tt_output_torch,
        ref_output,
        expected_pcc,
        "prepare_expert_weights_single_device",
    )
    assert passing, f"Single device prepare_expert_weights test failed. PCC: {pcc} < {expected_pcc}"
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
def test_gpt_oss_prepare_expert_weights_device_perf(mode, seq_len):
    """Device performance test for gpt_oss_prepare_expert_weights.

    This test measures device kernel duration and op-to-op latency using Tracy profiler.
    """
    # Batch size for decode_128 configuration
    batch_size = 128 if mode == "decode" else 1

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"gpt_oss_prepare_expert_weights_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_prepare_expert_weights.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_gpt_oss_prepare_expert_weights -k "{expr}"'

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
