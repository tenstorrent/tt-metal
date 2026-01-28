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
DEVICE_PERF_ENV_VAR = "DS_FUSED_ALL_GATHER_EMBEDDING_DEVICE_PERF"
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


def ds_fused_all_gather_embedding_reference(
    x: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """
    Reference implementation for AllGather in embedding module.

    The all_gather in embedding gathers across cluster_axis=0 (mesh rows) on dim=-1.
    Input per device: [1, 1, batch, per_device_hidden] where per_device_hidden = hidden_size/32
    Output per device: [1, 1, batch, per_row_hidden] where per_row_hidden = per_device_hidden * num_rows

    In the reference model (without tensor parallelism), this simulates gathering
    data from all rows by concatenating along the last dimension.

    Args:
        x: Input tensor of shape [1, 1, batch, per_device_hidden * num_rows]
           representing the full data that would be gathered from all rows
        num_rows: Number of mesh rows (4 for TG)

    Returns:
        Output tensor (same as input in reference model since we simulate full data)
    """
    # In reference, input already contains the full gathered data
    return x


def ds_fused_all_gather_embedding_ttnn(
    x: ttnn.Tensor,
    cfg: dict,
    ccl,
    persistent_output_buffer: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """
    TTNN implementation for AllGather in embedding module.

    This performs an all-gather operation across mesh rows (cluster_axis=0)
    to collect embedding data from all rows after the embedding lookup.

    Input per device: [1, 1, batch, 224] (224 = hidden_size/32)
    Output per device: [1, 1, batch, 896] (896 = 224 * 4 rows)

    Args:
        x: Input tensor sharded across devices
        cfg: Configuration dictionary containing all_gather config
        ccl: CCL runtime object
        persistent_output_buffer: Optional persistent output for trace mode

    Returns:
        Output tensor after all-gather
    """
    runtime_args = dict(ccl.populate_all_gather_runtime_args(cfg["all_gather"]))

    # Normalize negative dims (e.g., -1 -> last dimension) to avoid shape-check failures in C++.
    if "dim" in runtime_args and isinstance(runtime_args["dim"], int) and runtime_args["dim"] < 0:
        runtime_args["dim"] = runtime_args["dim"] % len(x.shape)

    # Handle persistent output buffer for trace mode
    if persistent_output_buffer is not None:
        if "mesh_device" in runtime_args:
            runtime_args["persistent_output_tensor"] = persistent_output_buffer
        else:
            runtime_args["persistent_output_buffer"] = persistent_output_buffer

    x = ttnn.experimental.all_gather_async(x, **runtime_args)
    return x


def _run_ds_fused_all_gather_embedding_test(
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
    ccl,
    step_prefix: str,
):
    # Log run configuration for superset
    _log_run_mode(mode, trace_mode, program_cache_enabled, seq_len)

    # Log config for verification
    logger.info(f"=== ALL_GATHER EMBEDDING OP CONFIG VERIFICATION ===")
    logger.info(f"Input shape: {tt_input.shape}")
    logger.info(f"Input memory_config: {tt_input.memory_config()}")
    logger.info(f"AllGather config: {run_config['all_gather']}")
    logger.info(f"=== END CONFIG VERIFICATION ===")

    tt_output = ds_fused_all_gather_embedding_ttnn(tt_input, run_config, ccl)

    # After all_gather on cluster_axis=0, data is gathered across rows
    # Each device now has [1, 1, batch, 896] (896 = 224 * 4)
    # Take output from first device for comparison
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")
    logger.info(f"ref_output shape: {ref_output.shape}")

    pcc_value, max_abs_error = _compare_with_reference(
        tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = _get_int_env("DS_ALLGATHER_EMBEDDING_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = _get_int_env("DS_ALLGATHER_EMBEDDING_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn(*, persistent_output_buffer=None):
            return ds_fused_all_gather_embedding_ttnn(
                tt_input, run_config, ccl, persistent_output_buffer=persistent_output_buffer
            )

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

        def op_fn(*, persistent_output_buffer=None):
            return ds_fused_all_gather_embedding_ttnn(
                tt_input, run_config, ccl, persistent_output_buffer=persistent_output_buffer
            )

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            # Trace mode: use a persistent output buffer
            persistent_output = op_fn()
            ttnn.synchronize_device(mesh_device)
            _ = op_fn(persistent_output_buffer=persistent_output)
            ttnn.synchronize_device(mesh_device)

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            op_fn(persistent_output_buffer=persistent_output)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.deallocate(persistent_output)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                output = op_fn()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
            signpost("stop")


def _build_all_gather_embedding_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    mode: str,
    seq_len: int,
):
    from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D

    # Generate random embedding state dict for weight config
    embedding_state_dict = {"weight": torch.randn(hf_config.vocab_size, hf_config.hidden_size, dtype=torch.float32)}

    weight_config = get_test_weight_config(
        Embedding1D,
        hf_config,
        (embedding_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(Embedding1D, mode, hf_config, mesh_device)
    model_state = Embedding1D.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    batch_size = USERS_PER_ROW if mode == "decode" else 1
    num_rows, num_cols = mesh_device.shape
    num_devices = mesh_device.get_num_devices()

    # Per-device hidden dimension
    # hidden_size is split across all 32 devices
    per_device_hidden = even_int_div(hf_config.hidden_size, num_devices)  # 224 = 7168/32

    # After all_gather on cluster_axis=0, the hidden dimension is gathered across rows
    # So output per device has: per_device_hidden * num_rows = 224 * 4 = 896
    per_row_hidden = per_device_hidden * num_rows  # 896

    # Input shape for embedding all_gather: [1, 1, batch_or_seq, per_device_hidden]
    # This is the shape after typecast and unsqueeze in embedding forward
    if mode == "decode":
        input_seq_len = batch_size  # In decode, seq_len dimension is batch_size
    else:
        input_seq_len = seq_len

    # Create input tensor that simulates output from typecast
    # Shape: [1, 1, input_seq_len, per_device_hidden] replicated across all devices
    # but each device has different data (sharded across devices)
    # For simplicity, we create the full gathered data and then shard it

    # Full data shape that would be gathered: [1, 1, input_seq_len, hidden_size]
    # But we need to create per-device data: [1, 1, input_seq_len, per_device_hidden]
    # After all_gather across rows, each device has: [1, 1, input_seq_len, per_row_hidden]

    # Create torch input representing what each device would have before all_gather
    # Each device (r, c) has a shard of the embedding output
    # For cluster_axis=0 gather, devices in the same column gather from all rows

    # Simulate the sharded input: [1, 1, input_seq_len, per_device_hidden] per device
    # To create a consistent test, we generate the full output and derive input from it

    # Generate full reference output that would result from all_gather
    # Shape: [1, 1, input_seq_len, per_row_hidden]
    torch_full_output = torch.randn(1, 1, input_seq_len, per_row_hidden, dtype=torch.bfloat16)

    # Reference output is the full gathered data
    ref_output = torch_full_output

    # Create input by taking slices that correspond to each row
    # Row r contributes: torch_full_output[:, :, :, r*per_device_hidden:(r+1)*per_device_hidden]
    # For the test, we replicate this pattern across the mesh

    # Create input tensor with shape [num_rows, 1, input_seq_len, per_device_hidden]
    # where each row r has the slice from the full output
    torch_input_per_row = torch.zeros(num_rows, 1, input_seq_len, per_device_hidden, dtype=torch.bfloat16)
    for r in range(num_rows):
        torch_input_per_row[r] = torch_full_output[:, :, :, r * per_device_hidden : (r + 1) * per_device_hidden]

    # Shard this across the mesh: rows get different data, columns get replicated
    # ShardTensor2dMesh with dims=(0, None) would shard on row but we need special handling
    # Instead, replicate full input to all devices, but each row's devices have different data

    # For the test to work with all_gather_async on cluster_axis=0:
    # - Devices in row r should have torch_input_per_row[r]
    # - This requires sharding on dim 0 across rows, replicating across columns

    tt_input = ttnn.from_torch(
        torch_input_per_row,
        device=mesh_device,
        # Shard dim 0 across rows (4 rows), replicate across columns (8 cols)
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    return run_config, tt_input, ref_output, batch_size, input_seq_len


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
        # Trace mode: use a persistent output buffer
        persistent_output = op_fn()
        ttnn.synchronize_device(mesh_device)
        _ = op_fn(persistent_output_buffer=persistent_output)
        ttnn.synchronize_device(mesh_device)

        logger.info("Capturing trace for perf...")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        op_fn(persistent_output_buffer=persistent_output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info("Trace captured. Replaying warmup...")

        for _ in range(warmup_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)

        logger.info("Warmup done. Replaying measured iterations...")
        profiler.clear()
        profiler.start("ds_fused_all_gather_embedding_perf")
        for _ in range(measure_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
        profiler.end("ds_fused_all_gather_embedding_perf", PERF_CNT=measure_iters)
        logger.info("Measured iterations done. Releasing trace...")
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(persistent_output)
        return profiler.get("ds_fused_all_gather_embedding_perf") * 1e6

    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("ds_fused_all_gather_embedding_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("ds_fused_all_gather_embedding_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_fused_all_gather_embedding_perf") * 1e6


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
                f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices {missing_devices}"
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


def _skip_single_device_ccl():
    pytest.skip("Single-device test is not applicable because ds_fused_all_gather_embedding includes CCL ops.")


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 32, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 512, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 1024, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 2048, 0.9999, 0.2, 0.2, 0.0),
    ],
)
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
def test_ds_fused_all_gather_embedding(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    program_cache_enabled,
    trace_mode,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
):
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

    run_config, tt_input, ref_output, batch_size, original_seq_len = _build_all_gather_embedding_inputs(
        mesh_device,
        hf_config,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        mode,
        seq_len,
    )
    _run_ds_fused_all_gather_embedding_test(
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
        ccl,
        f"ds_fused_all_gather_embedding_{mode}_seq{seq_len}",
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 32, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 512, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 1024, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 2048, 0.9999, 0.2, 0.2, 0.0),
    ],
)
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
def test_ds_fused_all_gather_embedding_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    program_cache_enabled,
    trace_mode,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    _skip_single_device_ccl()


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
def test_ds_fused_all_gather_embedding_device_perf(mode, seq_len):
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
    step_name = f"ds_fused_all_gather_embedding_device_perf_{mode}_seq{seq_len}"
    test_path = (
        "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/all_gather/test_ds_fused_all_gather_embedding.py"
    )
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_ds_fused_all_gather_embedding -k "{expr}"'

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
def test_ds_fused_all_gather_embedding_single_device_device_perf(mode, seq_len):
    _skip_single_device_ccl()


if __name__ == "__main__":
    pytest.main([__file__])
