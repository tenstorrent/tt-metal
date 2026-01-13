# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from collections import defaultdict
from typing import Literal

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.demos.deepseek_v3.tt.mlp.mlp import MLP
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_FUSED_REDUCE_SCATTER_POST_FF2_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 0.0, "op_to_op": 0.0},  # TODO: set real targets
    ("prefill", 128): {"kernel": 0.0, "op_to_op": 0.0},
    ("prefill", 1024): {"kernel": 0.0, "op_to_op": 0.0},
    ("prefill", 8192): {"kernel": 0.0, "op_to_op": 0.0},
}


def _get_int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as e:
        raise ValueError(f"Env var {name} must be an int, got {val!r}") from e


def ds_fused_reduce_scatter_post_ff2_reference(
    x: torch.Tensor, mesh_width: int, mode: Literal["decode", "prefill"]
) -> torch.Tensor:
    """
    Reference implementation for ReduceScatter_post_ff2.

    In the distributed model, reduce_scatter:
    1. Takes input from each device which is a partial product from w2 linear
    2. Performs a reduction (sum) across the mesh_width devices
    3. Scatters the result so each device gets 1/mesh_width of the output

    For the test, the input tensor `x` represents partial products from each device
    that are concatenated along the last dimension (simulating what all devices have).
    Each device originally has [..., dim], and there are mesh_width devices.
    So x has shape [..., dim * mesh_width].

    The reduce_scatter operation:
    1. Sums all partial products element-wise (each is [..., dim])
    2. Scatters the result: each device gets [..., dim / mesh_width]

    For the reference, we need to match the first device's output:
    - Sum: x[..., :dim] + x[..., dim:2*dim] + ... = summed_result [..., dim]
    - First device gets: summed_result[..., :dim/mesh_width]

    Args:
        x: Input tensor of shape [num_layers, seq_len, batch, dim * mesh_width] for decode
           or [num_layers, batch, seq_len, dim * mesh_width] for prefill.
           This represents the concatenated partial products from all mesh_width devices.
        mesh_width: Number of devices in the mesh width (typically 8 for TG).
        mode: "decode" or "prefill" (unused but kept for API consistency).

    Returns:
        Output tensor representing what the first device has after reduce_scatter.
        Shape is [num_layers, seq_len, batch, dim / mesh_width] for decode
        or [num_layers, batch, seq_len, dim / mesh_width] for prefill.
    """
    # Get the dimensions
    last_dim = x.shape[-1]
    per_device_input_dim = last_dim // mesh_width  # dim - what each device has as input
    per_device_output_dim = per_device_input_dim // mesh_width  # dim / mesh_width - output per device

    # Reshape to separate device contributions: [..., mesh_width, per_device_input_dim]
    shape_prefix = x.shape[:-1]
    x_reshaped = x.reshape(*shape_prefix, mesh_width, per_device_input_dim)

    # Sum across devices (dim=-2) to get the reduced result: [..., per_device_input_dim]
    x_reduced = x_reshaped.sum(dim=-2)

    # Scatter: each device gets 1/mesh_width of the reduced result
    # First device gets the first chunk: [..., per_device_output_dim]
    x_scattered = x_reduced[..., :per_device_output_dim]

    return x_scattered


def ds_fused_reduce_scatter_post_ff2_ttnn(
    x: ttnn.Tensor,
    cfg: dict,
    ccl,
    mode: Literal["decode", "prefill"],
    persistent_output_buffer: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """
    TTNN implementation for ReduceScatter_post_ff2.

    This performs the reduce_scatter operation after the w2 (down projection) linear layer.
    For decode mode, it first converts to DRAM memory config before the reduce_scatter.

    Args:
        x: Input tensor (output of w2 linear layer)
        cfg: Configuration dictionary containing reduce_scatter config
        ccl: CCL runtime object
        mode: "decode" or "prefill"
        persistent_output_buffer: Optional pre-allocated output buffer for trace mode

    Returns:
        Output tensor after reduce_scatter
    """
    if mode == "decode":
        # Decode mode: convert to DRAM first (as in MLP.forward_decode)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    # Get runtime args from CCL
    runtime_args = dict(ccl.populate_reduce_scatter_runtime_args(cfg["reduce_scatter_async"]))

    # Normalize negative dims
    if "dim" in runtime_args and isinstance(runtime_args["dim"], int) and runtime_args["dim"] < 0:
        runtime_args["dim"] = runtime_args["dim"] % len(x.shape)

    # Handle persistent output buffer for trace mode
    if persistent_output_buffer is not None:
        runtime_args["persistent_output_buffers"] = persistent_output_buffer

    output = ttnn.experimental.reduce_scatter_minimal_async(x, **runtime_args)
    return output


def _compare_with_reference(
    tt_output: torch.Tensor, ref_output: torch.Tensor, expected_pcc: float, atol: float, rtol: float
):
    passing, pcc = comp_pcc(ref_output, tt_output, expected_pcc)
    logger.info(f"PCC: {pcc}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"
    try:
        torch.testing.assert_close(tt_output, ref_output, rtol=rtol, atol=atol)
    except AssertionError as e:
        logger.warning(f"assert_close failed but PCC passed: {e}")


def _measure_perf_us(
    mesh_device: ttnn.MeshDevice, op_fn, warmup_iters: int, measure_iters: int, trace_mode: bool = False
) -> float:
    ttnn.synchronize_device(mesh_device)
    if trace_mode:
        # Trace mode: use a persistent output buffer and avoid deallocate inside trace capture.
        persistent_output = op_fn()
        ttnn.synchronize_device(mesh_device)
        # Warm up the persistent-buffer variant before capture
        _ = op_fn(persistent_output_buffer=persistent_output)
        ttnn.synchronize_device(mesh_device)

        logger.info("Capturing trace for perf…")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        op_fn(persistent_output_buffer=persistent_output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info("Trace captured. Replaying warmup…")

        for _ in range(warmup_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)

        logger.info("Warmup done. Replaying measured iterations…")
        profiler.clear()
        profiler.start("ds_fused_reduce_scatter_post_ff2_perf")
        for _ in range(measure_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
        profiler.end("ds_fused_reduce_scatter_post_ff2_perf", PERF_CNT=measure_iters)
        logger.info("Measured iterations done. Releasing trace…")
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(persistent_output)
        return profiler.get("ds_fused_reduce_scatter_post_ff2_perf") * 1e6

    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("ds_fused_reduce_scatter_post_ff2_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("ds_fused_reduce_scatter_post_ff2_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_fused_reduce_scatter_post_ff2_perf") * 1e6


def _run_ds_fused_reduce_scatter_post_ff2_test(
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
    tt_output = ds_fused_reduce_scatter_post_ff2_ttnn(tt_input, run_config, ccl, mode)

    # Get output from the first device only for comparison with reference
    # (similar to how all_gather test does it)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    _compare_with_reference(tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol)

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = _get_int_env("DS_REDUCE_SCATTER_POST_FF2_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = _get_int_env("DS_REDUCE_SCATTER_POST_FF2_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn(*, persistent_output_buffer=None):
            return ds_fused_reduce_scatter_post_ff2_ttnn(
                tt_input, run_config, ccl, mode, persistent_output_buffer=persistent_output_buffer
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
            return ds_fused_reduce_scatter_post_ff2_ttnn(
                tt_input, run_config, ccl, mode, persistent_output_buffer=persistent_output_buffer
            )

        for _ in range(PERF_WARMUP_ITERS):
            output = op_fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
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


def _build_reduce_scatter_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    mode: str,
    seq_len: int,
):
    """Build inputs for reduce_scatter_post_ff2 test.

    The input to reduce_scatter is the output of the w2 linear layer (down projection).
    In the MLP, reduce_scatter takes partial products from each device and:
    1. Reduces (sums) across mesh columns
    2. Scatters the result across mesh columns

    For this test:
    - Create input tensor that will be sharded across the mesh
    - Each device gets a portion of the input
    - reduce_scatter sums contributions from all devices and scatters
    """
    # Get MLP config to get the reduce_scatter configuration
    weight_config = get_test_weight_config(
        MLP,
        hf_config,
        (None,) * mesh_device.shape[0],  # No actual weights needed for this test
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(MLP, mode, hf_config, mesh_device)
    model_state = {
        "mesh_device": mesh_device,
        "mesh_shape": mesh_device.shape,
        "ccl": ccl,
    }
    run_config = create_run_config(model_config, weight_config, model_state)

    batch_size = USERS_PER_ROW if mode == "decode" else 1
    num_layers = mesh_device.shape[0]
    mesh_width = mesh_device.shape[1]
    dim = hf_config.hidden_size

    # Create input tensor with shape [num_layers, ..., dim]
    # This will be sharded: each device gets [1, ..., dim/mesh_width] after sharding on dims=(0, -1)
    if mode == "decode":
        torch_input = torch.randn(num_layers, seq_len, batch_size, dim, dtype=torch.bfloat16)
    else:
        torch_input = torch.randn(num_layers, batch_size, seq_len, dim, dtype=torch.bfloat16)

    # Shard across mesh devices with dims=(0, -1)
    # Each device (row=r, col=c) gets portion:
    # - Layer: layers[r]
    # - Dim: dim[c * dim/mesh_width : (c+1) * dim/mesh_width]
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Reference output for the first device (row=0, col=0):
    # reduce_scatter on cluster_axis=1 (mesh columns) with dim=3 means:
    # 1. Each device in a row has shape [1, seq, batch, dim/mesh_width]
    # 2. reduce: sum all mesh_width devices' tensors elementwise -> [1, seq, batch, dim/mesh_width]
    # 3. scatter along dim 3: divide into mesh_width chunks, each device gets one chunk
    #    -> output shape [1, seq, batch, dim/mesh_width/mesh_width]
    #
    # For the reference simulation:
    # - First layer data: torch_input[:1] has shape [1, seq, batch, dim]
    # - This represents all column devices' data concatenated on last dim
    # - Reshape to [1, seq, batch, mesh_width, dim/mesh_width] to separate contributions
    # - Sum across dim -2 (devices) -> [1, seq, batch, dim/mesh_width]
    # - Scatter: first device gets first chunk -> [1, seq, batch, dim/mesh_width/mesh_width]

    per_device_dim = dim // mesh_width
    per_device_output_dim = per_device_dim // mesh_width

    # Get first layer and reshape to separate device contributions
    first_layer = torch_input[:1]  # [1, seq, batch, dim]
    shape_prefix = first_layer.shape[:-1]  # [1, seq, batch]
    first_layer_reshaped = first_layer.reshape(*shape_prefix, mesh_width, per_device_dim)

    # Sum across devices (simulate reduce)
    reduced = first_layer_reshaped.sum(dim=-2)  # [1, seq, batch, per_device_dim]

    # Scatter: first device gets first chunk
    ref_output = reduced[..., :per_device_output_dim]  # [1, seq, batch, per_device_output_dim]

    return run_config, tt_input, ref_output, batch_size


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


def _skip_single_device_ccl():
    pytest.skip("Single-device test is not applicable because ds_fused_reduce_scatter_post_ff2 includes CCL ops.")


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 1024, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 8192, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 131072, 0.9999, 0.2, 0.2, 0.0),
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
def test_ds_fused_reduce_scatter_post_ff2(
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
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    run_config, tt_input, ref_output, batch_size = _build_reduce_scatter_inputs(
        mesh_device,
        hf_config,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        mode,
        seq_len,
    )
    _run_ds_fused_reduce_scatter_post_ff2_test(
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
        f"ds_fused_reduce_scatter_post_ff2_{mode}_seq{seq_len}",
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 1024, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 8192, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 131072, 0.9999, 0.2, 0.2, 0.0),
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
def test_ds_fused_reduce_scatter_post_ff2_single_device(
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
        ("decode", 1),
        ("prefill", 128),
        ("prefill", 1024),
        ("prefill", 8192),
        ("prefill", 131072),
    ],
)
def test_ds_fused_reduce_scatter_post_ff2_device_perf(mode, seq_len):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_fused_reduce_scatter_post_ff2_device_perf_{mode}_seq{seq_len}"
    test_path = (
        "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/reduce_scatter/test_ds_fused_reduce_scatter_post_ff2.py"
    )
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_ds_fused_reduce_scatter_post_ff2 -k "{expr}"'

    profiler.start("run")
    profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = _collect_device_perf(
        command,
        subdir="deepseek_v3_fused_ops_device_perf",
        warmup_iters=0,
        use_signposts=True,
    )
    os.environ.pop(DEVICE_PERF_ENV_VAR, None)
    profiler.end(step_name)
    profiler.end("run")

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
        profiler,
        0,
        step_name,
        "total_kernel_duration_us",
        total_kernel_us,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        step_name,
        "total_op_to_op_latency_us",
        total_op_to_op_us,
    )
    benchmark_data.save_partial_run_json(
        profiler,
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
        ("prefill", 8192),
        ("prefill", 131072),
    ],
)
def test_ds_fused_reduce_scatter_post_ff2_single_device_device_perf(mode, seq_len):
    _skip_single_device_ccl()


if __name__ == "__main__":
    pytest.main([__file__])
