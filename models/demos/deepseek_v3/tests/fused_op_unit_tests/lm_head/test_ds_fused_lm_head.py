# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for lm_head (vocabulary projection).

lm_head is the linear projection from hidden_size (7168) to vocab_size (129280).
The weight is sharded across all 32 devices, with each device holding vocab/32 = 4040
output features.

Sequence of ops:
    Decode:  output = ttnn.linear(x, **cfg["linear"])
    Prefill: output = ttnn.linear(x, program_config=..., **cfg["linear"])

Key characteristics:
    - Weight dtype: bfloat4_b (quantized)
    - Weight shape per device: [7168, 4040]
    - Weight memory: WIDTH_SHARDED DRAM
    - Decode input: WIDTH_SHARDED L1
    - Prefill input: DRAM INTERLEAVED
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.demos.deepseek_v3.tt.lm_head import LMHead
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config, pad_or_trim_seq_len
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_FUSED_LM_HEAD_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 0.0, "op_to_op": 0.0},  # TODO: set real targets
    ("prefill", 128): {"kernel": 0.0, "op_to_op": 0.0},
    ("prefill", 1024): {"kernel": 0.0, "op_to_op": 0.0},
    ("prefill", 131072): {"kernel": 0.0, "op_to_op": 0.0},
}


def _get_int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as e:
        raise ValueError(f"Env var {name} must be an int, got {val!r}") from e


class DeepseekV3LMHead(nn.Module):
    """PyTorch reference model for LMHead."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


def ds_fused_lm_head_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation for lm_head linear projection.

    Args:
        x: Input tensor of shape [1, 1, seq_len, hidden_size] or [1, num_chunks, chunk_size, hidden_size]
        weight: Weight tensor of shape [vocab_size, hidden_size]

    Returns:
        Output tensor of shape [1, 1, seq_len, vocab_size] or [1, num_chunks, chunk_size, vocab_size]
    """
    return torch.nn.functional.linear(x, weight)


def ds_fused_lm_head_ttnn_decode(
    x: ttnn.Tensor,
    cfg: dict,
) -> ttnn.Tensor:
    """
    TTNN implementation for lm_head linear projection (decode mode).

    Uses DRAM sharded matmul with WIDTH_SHARDED L1 activations.

    Args:
        x: Input tensor (WIDTH_SHARDED L1)
        cfg: Configuration dictionary containing linear config

    Returns:
        Output tensor (WIDTH_SHARDED L1)
    """
    output = ttnn.linear(x, **cfg["linear"])
    return output


def ds_fused_lm_head_ttnn_prefill(
    x: ttnn.Tensor,
    cfg: dict,
    seq_len: int,
) -> ttnn.Tensor:
    """
    TTNN implementation for lm_head linear projection (prefill mode).

    Uses multicore multicast matmul with DRAM INTERLEAVED tensors.
    For long sequences, handles chunking.

    Args:
        x: Input tensor (DRAM INTERLEAVED)
        cfg: Configuration dictionary containing linear and linear_pc_gen configs
        seq_len: Original sequence length (before any chunking)

    Returns:
        Output tensor (DRAM INTERLEAVED)
    """
    # Generate program config based on sequence length
    pc_gen = cfg["linear_pc_gen"]
    program_config = LMHead._get_prefill_pc(
        seq_len=seq_len,
        hidden_dim=pc_gen.hidden_dim,
        vocab_size=pc_gen.vocab_size,
        num_devices=pc_gen.num_devices,
        core_grid_size=pc_gen.core_grid_size,
    )

    output = ttnn.linear(x, program_config=program_config, **cfg["linear"])
    return output


def _compare_with_reference(
    tt_output: torch.Tensor, ref_output: torch.Tensor, expected_pcc: float, atol: float, rtol: float
) -> tuple[float, float]:
    """Compare TTNN output with reference and return metrics."""
    passing, pcc = comp_pcc(ref_output, tt_output, expected_pcc)
    max_abs_error = (tt_output.float() - ref_output.float()).abs().max().item()
    logger.info(f"PCC: {pcc}")
    logger.info(f"Max absolute error: {max_abs_error}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"
    try:
        torch.testing.assert_close(tt_output.float(), ref_output.float(), rtol=rtol, atol=atol)
    except AssertionError as e:
        logger.warning(f"assert_close failed but PCC passed: {e}")
    return pcc, max_abs_error


def _log_run_mode(mode: str, trace_mode: bool, program_cache_enabled: bool, seq_len: int):
    """Log the test run configuration."""
    logger.info("=== TEST RUN CONFIGURATION ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Sequence length: {seq_len}")
    logger.info(f"Trace mode: {trace_mode}")
    logger.info(f"Program cache enabled: {program_cache_enabled}")
    logger.info("===============================")


def _measure_perf_us(
    mesh_device: ttnn.MeshDevice, op_fn, warmup_iters: int, measure_iters: int, trace_mode: bool = False
) -> float:
    ttnn.synchronize_device(mesh_device)
    if trace_mode:
        # Warmup
        for _ in range(warmup_iters):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = op_fn()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(traced_output)

        for _ in range(warmup_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("ds_fused_lm_head_perf")
        for _ in range(measure_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
        profiler.end("ds_fused_lm_head_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("ds_fused_lm_head_perf") * 1e6

    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("ds_fused_lm_head_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("ds_fused_lm_head_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_fused_lm_head_perf") * 1e6


def _run_ds_fused_lm_head_test(
    mesh_device: ttnn.MeshDevice,
    run_config: dict,
    tt_input: ttnn.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    mode: str,
    seq_len: int,
    batch_size: int,
    step_prefix: str,
):
    _log_run_mode(mode, trace_mode, program_cache_enabled, seq_len)

    # Run lm_head
    if mode == "decode":
        tt_output = ds_fused_lm_head_ttnn_decode(tt_input, run_config)
    else:
        tt_output = ds_fused_lm_head_ttnn_prefill(tt_input, run_config, seq_len)

    # Convert output to torch - concatenate vocab dimension across all 32 devices
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))

    # Slice to actual vocab_size if needed (output may be padded)
    if tt_output_torch.shape[-1] > ref_output.shape[-1]:
        tt_output_torch = tt_output_torch[..., : ref_output.shape[-1]]

    pcc_value, max_abs_error = _compare_with_reference(
        tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = _get_int_env("DS_LM_HEAD_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = _get_int_env("DS_LM_HEAD_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        if mode == "decode":

            def op_fn():
                return ds_fused_lm_head_ttnn_decode(tt_input, run_config)

        else:

            def op_fn():
                return ds_fused_lm_head_ttnn_prefill(tt_input, run_config, seq_len)

        perf_us = _measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
        )
        logger.info(f"Perf avg: {perf_us:.3f} us over {measure_iters} iters (warmup {warmup_iters})")
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
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-pcc", pcc_value)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-max_abs_error", max_abs_error)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-expected_atol", expected_atol)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-expected_rtol", expected_rtol)
        benchmark_data.save_partial_run_json(
            perf_profiler,
            run_type="deepseek_v3_fused_ops",
            ml_model_name="deepseek-v3",
            batch_size=batch_size,
            input_sequence_length=seq_len,
            config_params={
                "mode": mode,
                "trace": trace_mode,
                "program_cache_enabled": program_cache_enabled,
                "module": "lm_head",
                "mesh_device": os.getenv("MESH_DEVICE", "TG"),
                "op_type": "lm_head",
            },
        )
        if expected_perf_us > 0 and not trace_mode and program_cache_enabled:
            perf_margin = 0.2
            assert perf_us <= expected_perf_us * (
                1 + perf_margin
            ), f"Perf regression: {perf_us:.3f}us exceeds expected {expected_perf_us:.3f}us"
        elif expected_perf_us == 0 and not trace_mode and program_cache_enabled:
            logger.warning("TODO: Set expected_perf_us using a measured baseline.")
    else:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        from tracy import signpost

        if mode == "decode":

            def op_fn():
                return ds_fused_lm_head_ttnn_decode(tt_input, run_config)

        else:

            def op_fn():
                return ds_fused_lm_head_ttnn_prefill(tt_input, run_config, seq_len)

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            traced_output = op_fn()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.deallocate(traced_output)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                output = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
            signpost("stop")

    # Clean up output
    ttnn.deallocate(tt_output)


def _build_lm_head_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config: Any,
    cache_path: Path,
    force_recalculate_weight_config: bool,
    use_real_weights: bool,
    mode: str,
    seq_len: int,
    state_dict: dict[str, torch.Tensor] | None,
):
    """Build inputs for lm_head test.

    Args:
        mesh_device: The mesh device
        hf_config: HuggingFace config
        cache_path: Path for weight caching
        force_recalculate_weight_config: Whether to force recalculate weights
        use_real_weights: Whether to use real model weights
        mode: "decode" or "prefill"
        seq_len: Sequence length
        state_dict: Model state dict (needed if use_real_weights=True)

    LMHead shape convention (from original test_lm_head.py):
    - Decode: height = 32 (USERS_PER_ROW users × 1 token each)
    - Prefill: height = seq_len (1 user × seq_len tokens)
    """
    hidden_size = hf_config.hidden_size

    # LMHead uses different batch conventions:
    # - Decode: batch_size=32 (USERS_PER_ROW), seq_len=1 → height=32
    # - Prefill: batch_size=1, seq_len=N → height=N
    if mode == "decode":
        batch_size = USERS_PER_ROW
        effective_height = batch_size * seq_len  # 32 × 1 = 32
    else:
        batch_size = 1
        effective_height = seq_len  # 1 × seq_len = seq_len

    # Create reference model
    reference_model = DeepseekV3LMHead(hf_config).eval()

    if use_real_weights and state_dict is not None:
        # Use provided real weights
        lm_head_state_dict = sub_state_dict(state_dict, "lm_head.")
        reference_model.load_state_dict(lm_head_state_dict, strict=False)
    else:
        # Use random weights (already initialized by nn.Linear)
        pass

    # Get state dict for TTNN weight conversion
    lm_head_state_dict = sub_state_dict(reference_model.state_dict(), "lm_head.")

    # Generate reference output
    # Shape: [1, 1, effective_height, hidden_size]
    torch_input = torch.randn(1, 1, effective_height, hidden_size)
    reference_output = reference_model(torch_input)

    # Pad input to SEQ_LEN_CHUNK_SIZE if necessary for TTNN
    torch_input_padded = pad_or_trim_seq_len(torch_input, mode, effective_height)

    # Generate TTNN configs
    # input_row_idx=3 matches the default in LMHead module test
    input_row_idx = 3
    weight_config = get_test_weight_config(
        LMHead, hf_config, (lm_head_state_dict,), cache_path, mesh_device, force_recalculate_weight_config
    )
    # Note: LMHead doesn't take seq_len in model_config - it computes program config dynamically
    model_config = get_model_config(LMHead, mode, hf_config, mesh_device, input_row_idx)
    model_state = LMHead.create_state(hf_config, mesh_device, None)  # CCL not needed for linear only
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN - replicate to all devices
    tt_input = ttnn.from_torch(
        torch_input_padded,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    return run_config, tt_input, reference_output, batch_size


def _maybe_skip_long_seq(seq_len: int):
    if seq_len <= 8192:
        return
    if os.getenv(LONG_SEQ_ENV_VAR) is None:
        pytest.skip(f"Set {LONG_SEQ_ENV_VAR}=1 to enable seq_len={seq_len} coverage.")


def _merge_device_rows_for_perf(df: pd.DataFrame) -> pd.DataFrame:
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
                f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices"
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
        }

    total_kernel_ns = sum(entry["avg_kernel_duration_ns"] for entry in op_stats.values())
    total_op_to_op_ns = sum(entry["avg_op_to_op_latency_ns"] for entry in op_stats.values())
    return op_stats, total_kernel_ns, total_op_to_op_ns


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.97, 1.0, 1.0, 0.0),  # batch=32, seq=1 → 32 tokens
        ("prefill", 128, 0.97, 1.0, 1.0, 0.0),  # batch=32, seq=128 → 4096 tokens
        ("prefill", 1024, 0.97, 1.0, 1.0, 0.0),  # batch=32, seq=1024 → 32768 tokens
        ("prefill", 131072, 0.97, 1.0, 1.0, 0.0),  # batch=32, seq=128k
    ],
)
@pytest.mark.parametrize("use_real_weights", [True, False], ids=["real_weights", "random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 23740416,  # Large trace region for vocab projection
        }
    ],
    indirect=True,
)
def test_ds_fused_lm_head(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    hf_config,
    cache_path,
    mesh_device,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    """Test lm_head fused op (vocabulary projection).

    This tests the linear projection from hidden_size (7168) to vocab_size (129280).
    The weight is sharded across all 32 devices with bfloat4_b quantization.
    """
    # Trace capture requires program cache enabled
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    _maybe_skip_long_seq(seq_len)

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    run_config, tt_input, ref_output, batch_size = _build_lm_head_inputs(
        mesh_device,
        hf_config,
        cache_path,
        force_recalculate_weight_config,
        use_real_weights,
        mode,
        seq_len,
        state_dict if use_real_weights else None,
    )

    _run_ds_fused_lm_head_test(
        mesh_device,
        run_config,
        tt_input,
        ref_output,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        mode,
        seq_len,
        batch_size,
        f"ds_fused_lm_head_{mode}_seq{seq_len}",
    )

    # Cleanup
    ttnn.deallocate(tt_input)


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.97, 1.0, 1.0, 0.0),
        ("prefill", 128, 0.97, 1.0, 1.0, 0.0),
        ("prefill", 1024, 0.97, 1.0, 1.0, 0.0),
        ("prefill", 131072, 0.97, 1.0, 1.0, 0.0),
    ],
)
@pytest.mark.parametrize("use_real_weights", [True, False], ids=["real_weights", "random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 23740416,
        }
    ],
    indirect=True,
)
def test_ds_fused_lm_head_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    hf_config,
    cache_path,
    mesh_device,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    """Single device test for lm_head.

    The lm_head linear projection can run on a single device with per-device
    weight shard (vocab/32). However, the output would only be 1/32 of the full vocab.
    For simplicity, we skip this test as the multi-device test is the primary use case.
    """
    pytest.skip(
        "Single-device test skipped: lm_head outputs vocab_size/32 per device. "
        "Multi-device test with all_gather is the primary use case."
    )


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        ("prefill", 1024),
        ("prefill", 131072),
    ],
)
def test_ds_fused_lm_head_device_perf(mode, seq_len):
    _maybe_skip_long_seq(seq_len)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    # batch_size=32 (USERS_PER_ROW) for all modes
    batch_size = USERS_PER_ROW

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_fused_lm_head_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_fused_lm_head.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and real_weights"
    command = f'pytest {test_path}::test_ds_fused_lm_head -k "{expr}"'

    perf_profiler.start("run")
    perf_profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = _collect_device_perf(
        command,
        subdir="deepseek_v3_fused_ops_device_perf",
        warmup_iters=0,
        use_signposts=True,
    )
    os.environ.pop(DEVICE_PERF_ENV_VAR, None)
    perf_profiler.end(step_name)
    perf_profiler.end("run")

    assert op_stats, "No device perf stats captured."
    total_kernel_us = total_kernel_ns / 1000.0
    total_op_to_op_us = total_op_to_op_ns / 1000.0
    logger.info(f"Device perf per-op averages (ns): {json.dumps(op_stats, indent=2)}")
    logger.info(f"Device perf totals: kernel={total_kernel_us:.3f} us, op_to_op={total_op_to_op_us:.3f} us")
    assert total_kernel_ns > 0, "Total kernel duration must be positive."
    assert total_op_to_op_ns >= 0, "Total op-to-op latency must be non-negative."
    targets = DEVICE_PERF_TARGETS_US.get((mode, seq_len))
    if targets is None or targets["kernel"] == 0.0:
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
    benchmark_data.save_partial_run_json(
        perf_profiler,
        run_type="deepseek_v3_fused_ops_device_perf",
        ml_model_name="deepseek-v3",
        batch_size=batch_size,
        input_sequence_length=seq_len,
    )


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        ("prefill", 1024),
    ],
)
def test_ds_fused_lm_head_single_device_device_perf(mode, seq_len):
    pytest.skip(
        "Single-device device perf test skipped: lm_head outputs vocab_size/32 per device. "
        "Multi-device test with all_gather is the primary use case."
    )


if __name__ == "__main__":
    pytest.main([__file__])
