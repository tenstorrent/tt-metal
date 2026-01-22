# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

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
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, even_int_div
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_FUSED_EMBEDDING_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 32): {"kernel": 0.0, "op_to_op": 0.0},  # TODO: Add theoretical targets
    ("prefill", 128): {"kernel": 0.0, "op_to_op": 0.0},  # TODO: Add theoretical targets
    ("prefill", 512): {"kernel": 0.0, "op_to_op": 0.0},  # TODO: Add theoretical targets
    ("prefill", 2048): {"kernel": 0.0, "op_to_op": 0.0},  # TODO: Add theoretical targets
}


def _get_int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as e:
        raise ValueError(f"Env var {name} must be an int, got {val!r}") from e


def ds_fused_embedding_reference(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation for the embedding fused op.

    This uses torch.nn.functional.embedding to look up embeddings.

    Args:
        input_ids: Input token IDs tensor [1, 1, seq_len] or [1, 1, batch_size]
        weight: Embedding weight tensor [vocab_size, hidden_size]

    Returns:
        Output tensor: [1, 1, seq_len, hidden_size] or [1, 1, batch_size, hidden_size]
    """
    # input_ids is [1, 1, seq_len], squeeze to get [seq_len]
    ids = input_ids.squeeze()
    # Embedding lookup
    embeddings = torch.nn.functional.embedding(ids, weight)
    # Reshape to match TTNN output: [1, 1, seq_len, hidden_size]
    return embeddings.unsqueeze(0).unsqueeze(0)


def ds_fused_embedding_ttnn(
    input_ids: ttnn.Tensor,
    cfg: dict,
    original_seq_len: int,
) -> ttnn.Tensor:
    """
    TTNN implementation for the embedding fused op.

    This performs: ttnn.embedding(x, **cfg["embedding"]) with optional padding.
    Matches lines 180-184 of embedding1d.py.

    Args:
        input_ids: Input token IDs tensor
        cfg: Configuration dictionary containing embedding config
        original_seq_len: Original sequence length before padding

    Returns:
        Output tensor after embedding lookup
    """
    if original_seq_len % ttnn.TILE_SIZE == 0:
        embeddings = ttnn.embedding(input_ids, **cfg["embedding"])
    else:
        x_padded = ttnn.pad(input_ids, [(0, 0), (0, 0), (0, ttnn.TILE_SIZE - original_seq_len % ttnn.TILE_SIZE)], 0)
        embeddings = ttnn.embedding(x_padded, **cfg["embedding"])
        ttnn.deallocate(x_padded)

    return embeddings


def _run_ds_fused_embedding_test(
    mesh_device: ttnn.MeshDevice,
    run_config: dict,
    tt_input_ids: ttnn.Tensor,
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
    original_seq_len: int,
):
    # Log run configuration for superset
    _log_run_mode(mode, trace_mode, program_cache_enabled, seq_len)

    # Log config for verification (Step 9 of AGENTS_GUIDE)
    logger.info(f"=== EMBEDDING OP CONFIG VERIFICATION ===")
    logger.info(f"Input input_ids shape: {tt_input_ids.shape}")
    logger.info(f"Input input_ids memory_config: {tt_input_ids.memory_config()}")
    logger.info(f"Embedding weight shape: {run_config['embedding']['weight'].shape}")
    logger.info(f"Embedding config memory_config: {run_config['embedding']['memory_config']}")
    logger.info(f"Embedding config layout: {run_config['embedding']['layout']}")
    logger.info(f"=== END CONFIG VERIFICATION ===")

    # Run the embedding op
    tt_output = ds_fused_embedding_ttnn(tt_input_ids, run_config, original_seq_len)

    # Collect output from all devices and compare
    # The embedding weight is sharded across ALL 32 devices (4x8 mesh)
    # Each device has [1, seq_len, hidden_size/32] = [1, seq_len, 224]
    # ConcatMesh2dToTensor with dims=(0, -1) gives:
    #   - Concatenate 8 columns on dim -1: [1, seq_len, 224*8=1792] per row
    #   - Stack 4 rows on dim 0: [4, seq_len, 1792]
    # To get full hidden_size, we need to reshape and concatenate rows
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Log output shape for debugging
    logger.info(f"tt_output_torch shape after mesh concat: {tt_output_torch.shape}")
    # Shape is [num_rows, seq_len, hidden_per_row] = [4, seq_len, 1792]

    # Trim to original seq_len first
    tt_output_torch = tt_output_torch[:, :original_seq_len, :]

    # Reconstruct full hidden dimension with correct ordering
    # The embedding weight was reshaped from [vocab_size, hidden_size] to [vocab_size, cols, rows, per_device]
    # and sharded with dims=(2, 1), meaning:
    #   - device(r, c) gets weight[:, c, r, :]
    #   - tt[r, seq, c*per_device:(c+1)*per_device] = device(r, c) output
    # To reconstruct original hidden order:
    #   - original h = c*(rows*per_device) + r*per_device + local_h
    num_rows, seq_dim, hidden_per_row = tt_output_torch.shape
    num_cols = mesh_device.shape[1]  # 8 columns
    per_device_hidden = hidden_per_row // num_cols  # 224 typically

    # [4, seq, 1792] -> [4, seq, 8, 224]
    tt_output_torch = tt_output_torch.reshape(num_rows, seq_dim, num_cols, per_device_hidden)
    # [4, seq, 8, 224] -> [seq, 8, 4, 224] to match original reshape order
    tt_output_torch = tt_output_torch.permute(1, 2, 0, 3)
    # [seq, 8, 4, 224] -> [seq, 7168]
    tt_output_torch = tt_output_torch.reshape(seq_dim, -1)
    # [seq, 7168] -> [1, seq, 7168]
    tt_output_torch = tt_output_torch.unsqueeze(0)

    # Unsqueeze to match reference output shape [1, 1, seq_len, hidden_size]
    tt_output_torch = tt_output_torch.unsqueeze(1)
    logger.info(f"tt_output_torch final shape: {tt_output_torch.shape}")

    pcc_value, max_abs_error = _compare_with_reference(
        tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = _get_int_env("DS_FUSED_EMBEDDING_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = _get_int_env("DS_FUSED_EMBEDDING_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn():
            return ds_fused_embedding_ttnn(tt_input_ids, run_config, original_seq_len)

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
        # Log PCC and ATOL metrics to superset
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-pcc", pcc_value)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-max_abs_error", max_abs_error)
        benchmark_data.save_partial_run_json(
            perf_profiler,
            run_type="deepseek_v3_fused_ops",
            ml_model_name="deepseek-v3",
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
    else:
        logger.info("Skipping e2e perf measurement during device-perf profiling.")
        from tracy import signpost

        def op_fn():
            return ds_fused_embedding_ttnn(tt_input_ids, run_config, original_seq_len)

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            output = op_fn()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.deallocate(output)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                output = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
            signpost("stop")


def _build_embedding_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    mode: str,
    seq_len: int,
    use_real_weights: bool,
    state_dict: dict = None,
):
    from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D

    # For real weights, we need the state dict
    # For random weights, we generate a random embedding weight tensor
    if use_real_weights:
        assert state_dict is not None, "state_dict required for real weights"
        embedding_state_dict = state_dict
    else:
        # Generate random embedding weights matching the expected shape
        embedding_state_dict = {"weight": torch.randn(hf_config.vocab_size, hf_config.hidden_size, dtype=torch.float32)}
    state_dicts = (embedding_state_dict,)

    weight_config = get_test_weight_config(
        Embedding1D,
        hf_config,
        state_dicts,
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(Embedding1D, mode, hf_config, mesh_device)
    model_state = Embedding1D.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    batch_size = USERS_PER_ROW if mode == "decode" else 1
    num_layers = mesh_device.shape[0]

    # Get dimensions from config
    vocab_size = hf_config.vocab_size
    hidden_size = hf_config.hidden_size

    # Input shape: [1, 1, seq_len] for token IDs
    if mode == "decode":
        # For decode, seq_len is actually batch_size
        input_seq_len = batch_size
    else:
        input_seq_len = seq_len

    # Generate random token IDs
    torch_input_ids = torch.randint(0, vocab_size, (1, 1, input_seq_len), dtype=torch.int64)

    # For reference, use the same weight that was passed to the module
    torch_weight = embedding_state_dict["weight"].float()

    # Compute reference output
    ref_output = ds_fused_embedding_reference(torch_input_ids, torch_weight)

    # Convert input IDs to TTNN tensor - replicate across all devices
    tt_input_ids = ttnn.from_torch(
        torch_input_ids,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    return run_config, tt_input_ids, ref_output, batch_size, input_seq_len


def _maybe_skip_long_seq(seq_len: int):
    if seq_len <= 2048:
        return
    if os.getenv(LONG_SEQ_ENV_VAR) is None:
        pytest.skip(f"Set {LONG_SEQ_ENV_VAR}=1 to enable seq_len={seq_len} coverage.")


def _compare_with_reference(
    tt_output: torch.Tensor, ref_output: torch.Tensor, expected_pcc: float, atol: float, rtol: float
) -> tuple[float, float]:
    """Compare TTNN output with reference and return metrics.

    Returns:
        Tuple of (pcc_value, max_abs_error) for logging to superset.
    """
    passing, pcc = comp_pcc(ref_output, tt_output, expected_pcc)
    max_abs_error = (tt_output.float() - ref_output.float()).abs().max().item()
    logger.info(f"PCC: {pcc}")
    logger.info(f"Max absolute error: {max_abs_error}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"
    torch.testing.assert_close(tt_output.float(), ref_output.float(), rtol=rtol, atol=atol)
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
        # Warmup in eager mode first to compile
        for _ in range(warmup_iters):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = op_fn()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        # Warmup with trace
        for _ in range(warmup_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)

        # Measure with trace
        profiler.clear()
        profiler.start("ds_fused_embedding_perf")
        for _ in range(measure_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
        profiler.end("ds_fused_embedding_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(output)
        return profiler.get("ds_fused_embedding_perf") * 1e6

    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("ds_fused_embedding_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("ds_fused_embedding_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_fused_embedding_perf") * 1e6


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
        # PCC ~0.99 is acceptable for embedding with bfloat16
        ("decode", 32, 0.99, 0.1, 0.1, 0.0),
        ("prefill", 128, 0.99, 0.1, 0.1, 0.0),
        ("prefill", 512, 0.99, 0.1, 0.1, 0.0),
        ("prefill", 1024, 0.99, 0.1, 0.1, 0.0),
        ("prefill", 2048, 0.99, 0.1, 0.1, 0.0),
    ],
)
@pytest.mark.parametrize("use_real_weights", [True, False], ids=["real_weights", "random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 2967552,
        }
    ],
    indirect=True,
)
def test_ds_fused_embedding(
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
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    # Trace capture replays pre-compiled binaries. When program cache is disabled, ops may
    # trigger compilation/program writes during capture, which is forbidden and can TT_FATAL.
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if mode == "decode":
        # For decode mode, seq_len represents batch_size (32 for USERS_PER_ROW)
        assert seq_len == 32, "Decode mode uses batch_size=32 (USERS_PER_ROW)"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # For real weights, get the embedding state dict
    embedding_state_dict = None
    if use_real_weights:
        from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict

        embedding_state_dict = sub_state_dict(state_dict, "model.embed_tokens.")

    run_config, tt_input_ids, ref_output, batch_size, original_seq_len = _build_embedding_inputs(
        mesh_device,
        hf_config,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        mode,
        seq_len,
        use_real_weights,
        embedding_state_dict,
    )
    _run_ds_fused_embedding_test(
        mesh_device,
        run_config,
        tt_input_ids,
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
        f"ds_fused_embedding_{mode}_seq{seq_len}",
        original_seq_len,
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # Single device tests - embedding doesn't have CCL so this works
        ("decode", 32, 0.99, 0.1, 0.1, 0.0),
        ("prefill", 128, 0.99, 0.1, 0.1, 0.0),
        ("prefill", 512, 0.99, 0.1, 0.1, 0.0),
        ("prefill", 1024, 0.99, 0.1, 0.1, 0.0),
        ("prefill", 2048, 0.99, 0.1, 0.1, 0.0),
    ],
)
@pytest.mark.parametrize("use_real_weights", [True, False], ids=["real_weights", "random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 2967552,
        }
    ],
    indirect=True,
)
def test_ds_fused_embedding_single_device(
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
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    """
    Single device test for the embedding fused op.

    This test runs on a single device from the mesh. The embedding op itself
    doesn't use CCL, so we can test just the embedding lookup on a single device.
    """
    # Trace capture replays pre-compiled binaries. When program cache is disabled, ops may
    # trigger compilation/program writes during capture, which is forbidden and can TT_FATAL.
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if mode == "decode":
        assert seq_len == 32, "Decode mode uses batch_size=32 (USERS_PER_ROW)"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # Get single device from mesh
    single_device = mesh_device.get_device(0)

    # Get dimensions
    vocab_size = hf_config.vocab_size
    hidden_size = hf_config.hidden_size
    _, mesh_width = mesh_device.shape
    per_device_hidden_size = even_int_div(hidden_size, mesh_device.get_num_devices())

    batch_size = USERS_PER_ROW if mode == "decode" else 1

    if mode == "decode":
        input_seq_len = batch_size
    else:
        input_seq_len = seq_len

    # Generate random token IDs
    torch_input_ids = torch.randint(0, vocab_size, (1, 1, input_seq_len), dtype=torch.int64)

    # For reference, use per-device weight slice
    if use_real_weights:
        from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict

        embedding_state_dict = sub_state_dict(state_dict, "model.embed_tokens.")
        torch_weight = embedding_state_dict["weight"][:, :per_device_hidden_size].float()
    else:
        torch_weight = torch.randn(vocab_size, per_device_hidden_size, dtype=torch.float32)

    # Compute reference output (per-device slice)
    ref_output = ds_fused_embedding_reference(torch_input_ids, torch_weight)

    # Convert input IDs to TTNN tensor on single device
    tt_input_ids = ttnn.from_torch(
        torch_input_ids,
        device=single_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Convert weight to TTNN tensor on single device
    # Weight shape should be [1, 1, vocab_size, per_device_hidden_size] to match the module config
    torch_weight_4d = torch_weight.unsqueeze(0).unsqueeze(0)
    tt_weight = ttnn.from_torch(
        torch_weight_4d,
        device=single_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Pad input if necessary
    if input_seq_len % ttnn.TILE_SIZE == 0:
        tt_output = ttnn.embedding(
            tt_input_ids, tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
    else:
        x_padded = ttnn.pad(tt_input_ids, [(0, 0), (0, 0), (0, ttnn.TILE_SIZE - input_seq_len % ttnn.TILE_SIZE)], 0)
        tt_output = ttnn.embedding(x_padded, tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_padded)

    # Convert output to torch
    tt_output_torch = ttnn.to_torch(tt_output)[:, :input_seq_len, :]

    # Reshape to match reference [1, 1, seq_len, hidden_size]
    tt_output_torch = tt_output_torch.unsqueeze(0)

    # Compare with reference
    _compare_with_reference(tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol)


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 32),
        ("prefill", 128),
        ("prefill", 512),
        ("prefill", 1024),
        ("prefill", 2048),
    ],
)
def test_ds_fused_embedding_device_perf(mode, seq_len):
    if mode == "decode":
        assert seq_len == 32, "Decode mode uses batch_size=32"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0]

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_fused_embedding_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_fused_embedding.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and real_weights"
    command = f'pytest {test_path}::test_ds_fused_embedding -k "{expr}"'

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
        ("decode", 32),
        ("prefill", 128),
        ("prefill", 512),
        ("prefill", 1024),
        ("prefill", 2048),
    ],
)
def test_ds_fused_embedding_single_device_device_perf(mode, seq_len):
    """
    Single device device performance test for the embedding fused op.
    """
    if mode == "decode":
        assert seq_len == 32, "Decode mode uses batch_size=32"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0]

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_fused_embedding_single_device_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_fused_embedding.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and real_weights"
    command = f'pytest {test_path}::test_ds_fused_embedding_single_device -k "{expr}"'

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


if __name__ == "__main__":
    pytest.main([__file__])
