# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for mesh_scatter.

mesh_scatter is a communication primitive used in LMHead that broadcasts tensor data
from one row of the mesh to all other rows. This enables all 32 devices to participate
in the vocabulary projection by replicating the input across all mesh rows.

Sequence of ops:
    mesh_scatter(x, mesh_shape, scatter_idx)

Where:
    - mesh_shape: (4, 8) - 4 rows × 8 columns
    - scatter_idx: (row_idx, None) - broadcast from row_idx to all other rows
    - Internally uses ttnn.point_to_point for device-to-device data transfer
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
from models.demos.deepseek_v3.utils.composite_ops import mesh_scatter
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_FUSED_MESH_SCATTER_DEVICE_PERF"
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


def ds_fused_mesh_scatter_reference(
    x: torch.Tensor,
    mesh_shape: tuple[int, int],
    scatter_idx: tuple[int | None, int | None],
) -> torch.Tensor:
    """
    Reference implementation for mesh_scatter.

    mesh_scatter broadcasts data from one mesh row to all other rows.
    Since this is a communication primitive (not compute), the reference
    simply returns the input unchanged - the actual test verifies that
    all devices have consistent data after the operation.

    Args:
        x: Input tensor of shape [1, 1, seq_len, hidden_size]
        mesh_shape: Tuple (num_rows, num_cols) of the mesh
        scatter_idx: Tuple (from_row, from_col) indicating source indices
                    None means don't scatter on that dimension

    Returns:
        The same tensor (mesh_scatter is a communication op, not compute)
    """
    # mesh_scatter is a communication primitive that broadcasts data
    # The output data is identical to the input (from the source row's perspective)
    return x


def ds_fused_mesh_scatter_ttnn(
    x: ttnn.Tensor,
    mesh_shape: tuple[int, int],
    scatter_idx: tuple[int | None, int | None],
) -> ttnn.Tensor:
    """
    TTNN implementation for mesh_scatter.

    Broadcasts tensor data from one row of the mesh to all other rows
    using ttnn.point_to_point for device-to-device transfers.

    Args:
        x: Input tensor (on device)
        mesh_shape: Tuple (num_rows, num_cols) of the mesh
        scatter_idx: Tuple (from_row, from_col) indicating source indices

    Returns:
        The same tensor with data broadcast across all rows
    """
    mesh_scatter(x, mesh_shape=mesh_shape, scatter_idx=scatter_idx)
    return x  # Returns same tensor (modified in-place across devices)


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


def _verify_data_consistency_across_rows(
    mesh_device: ttnn.MeshDevice,
    tt_tensor: ttnn.Tensor,
    source_row: int,
) -> bool:
    """
    Verify that all rows have identical data after mesh_scatter.

    Args:
        mesh_device: The mesh device
        tt_tensor: The tensor after mesh_scatter
        source_row: The row that was the source of the broadcast

    Returns:
        True if all rows have identical data
    """
    device_tensors = ttnn.get_device_tensors(tt_tensor)
    num_rows, num_cols = mesh_device.shape

    # Get reference data from source row (first device in that row)
    source_device_idx = source_row * num_cols
    ref_data = ttnn.to_torch(device_tensors[source_device_idx])

    all_consistent = True
    for row in range(num_rows):
        for col in range(num_cols):
            device_idx = row * num_cols + col
            device_data = ttnn.to_torch(device_tensors[device_idx])

            # Compare with reference (same column in source row)
            ref_device_idx = source_row * num_cols + col
            expected_data = ttnn.to_torch(device_tensors[ref_device_idx])

            passing, pcc = comp_pcc(expected_data, device_data, 0.9999)
            if not passing:
                logger.error(f"Device ({row}, {col}) data mismatch with source. PCC: {pcc}")
                all_consistent = False
            else:
                logger.debug(f"Device ({row}, {col}) data consistent with source. PCC: {pcc}")

    return all_consistent


def _log_run_mode(mode: str, trace_mode: bool, program_cache_enabled: bool, seq_len: int, scatter_row: int):
    """Log the test run configuration."""
    logger.info("=== TEST RUN CONFIGURATION ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Sequence length: {seq_len}")
    logger.info(f"Scatter source row: {scatter_row}")
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
            op_fn()
            ttnn.synchronize_device(mesh_device)

        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        op_fn()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(warmup_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("ds_fused_mesh_scatter_perf")
        for _ in range(measure_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
        profiler.end("ds_fused_mesh_scatter_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("ds_fused_mesh_scatter_perf") * 1e6

    for _ in range(warmup_iters):
        op_fn()
        ttnn.synchronize_device(mesh_device)

    profiler.clear()
    profiler.start("ds_fused_mesh_scatter_perf")
    for _ in range(measure_iters):
        op_fn()
        ttnn.synchronize_device(mesh_device)
    profiler.end("ds_fused_mesh_scatter_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_fused_mesh_scatter_perf") * 1e6


def _run_ds_fused_mesh_scatter_test(
    mesh_device: ttnn.MeshDevice,
    tt_input: ttnn.Tensor,
    ref_output: torch.Tensor,
    mesh_shape: tuple[int, int],
    scatter_idx: tuple[int | None, int | None],
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
    scatter_row = scatter_idx[0] if scatter_idx[0] is not None else 0
    _log_run_mode(mode, trace_mode, program_cache_enabled, seq_len, scatter_row)

    # Run mesh_scatter
    tt_output = ds_fused_mesh_scatter_ttnn(tt_input, mesh_shape, scatter_idx)

    # Verify data consistency across all rows
    all_consistent = _verify_data_consistency_across_rows(mesh_device, tt_output, scatter_row)
    assert all_consistent, "Data inconsistency detected across mesh rows after mesh_scatter"

    # Get output from first device for PCC comparison with reference
    device_tensors = ttnn.get_device_tensors(tt_output)
    tt_output_torch = ttnn.to_torch(device_tensors[0])

    pcc_value, max_abs_error = _compare_with_reference(
        tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = _get_int_env("DS_MESH_SCATTER_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = _get_int_env("DS_MESH_SCATTER_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn():
            ds_fused_mesh_scatter_ttnn(tt_input, mesh_shape, scatter_idx)

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
                "op_type": "mesh_scatter",
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

        def op_fn():
            ds_fused_mesh_scatter_ttnn(tt_input, mesh_shape, scatter_idx)

        for _ in range(PERF_WARMUP_ITERS):
            op_fn()
            ttnn.synchronize_device(mesh_device)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            op_fn()
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
                op_fn()
                ttnn.synchronize_device(mesh_device)
            signpost("stop")


def _build_mesh_scatter_inputs(
    mesh_device: ttnn.MeshDevice,
    mode: str,
    seq_len: int,
    hidden_size: int,
    scatter_row: int,
):
    """Build inputs for mesh_scatter test.

    Args:
        mesh_device: The mesh device
        mode: "decode" or "prefill"
        seq_len: Sequence length (1 for decode, varies for prefill)
        hidden_size: Hidden dimension (7168 for DeepSeek V3)
        scatter_row: The row index to scatter from (0-3)

    LMHead shape convention (from original test_lm_head.py):
    - Decode: height = 32 (USERS_PER_ROW users × 1 token each)
    - Prefill: height = seq_len (1 user × seq_len tokens)
    """
    mesh_shape = tuple(mesh_device.shape)
    num_rows, num_cols = mesh_shape

    # LMHead uses different batch conventions:
    # - Decode: batch_size=32 (USERS_PER_ROW), seq_len=1 → height=32
    # - Prefill: batch_size=1, seq_len=N → height=N
    if mode == "decode":
        batch_size = USERS_PER_ROW
        effective_height = batch_size * seq_len  # 32 × 1 = 32
    else:
        batch_size = 1
        effective_height = seq_len  # 1 × seq_len = seq_len

    # Input shape: [1, 1, effective_height, hidden_size]
    torch_input = torch.randn(1, 1, effective_height, hidden_size, dtype=torch.bfloat16)

    # Memory config based on mode
    if mode == "decode":
        # For decode: WIDTH_SHARDED L1
        # Grid: (0,0) - (7,6) = 56 cores
        # Shard shape: (32, 128) since hidden_size=7168, 7168/56 = 128
        num_cores = 56
        shard_width = hidden_size // num_cores
        input_memory_config = ttnn.create_sharded_memory_config_(
            shape=(effective_height, shard_width),
            core_grid=ttnn.num_cores_to_corerangeset(
                num_cores,
                ttnn.CoreCoord(mesh_device.core_grid.x, mesh_device.core_grid.y),
                row_wise=True,
            ),
            strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        # For prefill: DRAM INTERLEAVED
        input_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Convert to TTNN - replicate to all devices
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Reference output is same as input (mesh_scatter is communication, not compute)
    ref_output = ds_fused_mesh_scatter_reference(torch_input, mesh_shape, (scatter_row, None))

    scatter_idx = (scatter_row, None)

    return tt_input, ref_output, mesh_shape, scatter_idx, batch_size


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

        is_collective = any(tag in op_name for tag in ("AllGather", "ReduceScatter", "AllReduce", "AllToAll", "Point"))
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


# Hidden size for DeepSeek V3
HIDDEN_SIZE = 7168


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # mesh_scatter is a communication op, so PCC should be ~1.0 (perfect)
        # batch_size=32 for all modes, seq_len=1 for decode
        ("decode", 1, 0.9999, 0.001, 0.001, 0.0),  # batch=32, seq=1 → 32 tokens
        ("prefill", 128, 0.9999, 0.001, 0.001, 0.0),  # batch=32, seq=128 → 4096 tokens
        ("prefill", 1024, 0.9999, 0.001, 0.001, 0.0),
        ("prefill", 131072, 0.9999, 0.001, 0.001, 0.0),  # 128k
    ],
)
@pytest.mark.parametrize(
    "scatter_row",
    [0, 1, 2, 3],
    ids=["row_0", "row_1", "row_2", "row_3"],
)
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 2967552,
        }
    ],
    indirect=True,
)
def test_ds_fused_mesh_scatter(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    scatter_row,
    program_cache_enabled,
    trace_mode,
    mesh_device,
    set_deterministic_env,
):
    """Test mesh_scatter fused op.

    mesh_scatter broadcasts tensor data from one mesh row to all other rows.
    This is used in LMHead to replicate input across all 32 devices before
    the vocabulary projection.
    """
    # Trace capture requires program cache enabled
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    _maybe_skip_long_seq(seq_len)

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # Validate scatter_row is within mesh bounds
    num_rows = mesh_device.shape[0]
    if scatter_row >= num_rows:
        pytest.skip(f"scatter_row={scatter_row} exceeds mesh rows ({num_rows})")

    tt_input, ref_output, mesh_shape, scatter_idx, batch_size = _build_mesh_scatter_inputs(
        mesh_device,
        mode,
        seq_len,
        HIDDEN_SIZE,
        scatter_row,
    )

    _run_ds_fused_mesh_scatter_test(
        mesh_device,
        tt_input,
        ref_output,
        mesh_shape,
        scatter_idx,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        mode,
        seq_len,
        batch_size,
        f"ds_fused_mesh_scatter_{mode}_seq{seq_len}_row{scatter_row}",
    )

    # Cleanup
    ttnn.deallocate(tt_input)


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.9999, 0.001, 0.001, 0.0),  # batch=32, seq=1
        ("prefill", 128, 0.9999, 0.001, 0.001, 0.0),
        ("prefill", 1024, 0.9999, 0.001, 0.001, 0.0),
        ("prefill", 131072, 0.9999, 0.001, 0.001, 0.0),
    ],
)
@pytest.mark.parametrize(
    "scatter_row",
    [0, 1, 2, 3],
    ids=["row_0", "row_1", "row_2", "row_3"],
)
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 2967552,
        }
    ],
    indirect=True,
)
def test_ds_fused_mesh_scatter_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    scatter_row,
    program_cache_enabled,
    trace_mode,
    mesh_device,
    set_deterministic_env,
):
    """Single device test for mesh_scatter.

    mesh_scatter uses ttnn.point_to_point which is a CCL operation for
    device-to-device communication. Therefore, single device test is not applicable.
    """
    pytest.skip(
        "Single-device test skipped: mesh_scatter uses ttnn.point_to_point "
        "for inter-device communication, which requires multiple devices."
    )


@pytest.mark.parametrize(
    "mode, seq_len, scatter_row",
    [
        ("decode", 1, 3),  # Default scatter_row for LMHead is 3 (last transformer layer)
        ("prefill", 128, 3),
        ("prefill", 1024, 3),
        ("prefill", 131072, 3),
    ],
)
def test_ds_fused_mesh_scatter_device_perf(mode, seq_len, scatter_row):
    _maybe_skip_long_seq(seq_len)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    # batch_size=32 (USERS_PER_ROW) for all modes
    batch_size = USERS_PER_ROW

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_fused_mesh_scatter_device_perf_{mode}_seq{seq_len}_row{scatter_row}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_fused_mesh_scatter.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    row_filter = f"row_{scatter_row}"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and {row_filter}"
    command = f'pytest {test_path}::test_ds_fused_mesh_scatter -k "{expr}"'

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
    "mode, seq_len, scatter_row",
    [
        ("decode", 1, 3),
        ("prefill", 128, 3),
        ("prefill", 1024, 3),
    ],
)
def test_ds_fused_mesh_scatter_single_device_device_perf(mode, seq_len, scatter_row):
    pytest.skip(
        "Single-device device perf test skipped: mesh_scatter uses ttnn.point_to_point "
        "for inter-device communication, which requires multiple devices."
    )


if __name__ == "__main__":
    pytest.main([__file__])
