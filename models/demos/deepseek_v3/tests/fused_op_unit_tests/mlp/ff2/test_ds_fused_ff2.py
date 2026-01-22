# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP
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
DEVICE_PERF_ENV_VAR = "DS_FUSED_FF2_DEVICE_PERF"
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


@dataclass
class FusedWeights:
    w2: torch.Tensor  # down_proj weight


def ds_fused_ff2_reference(x: torch.Tensor, weights: FusedWeights, mode: Literal["decode", "prefill"]) -> torch.Tensor:
    """
    Reference implementation for FF2 fused op (down projection).

    This is the down projection operation in the MLP that projects
    from intermediate_size back to hidden_size.

    Args:
        x: Input tensor from the mul operation (activated).
           Shape is [num_layers, 1, batch_size, intermediate_size] for decode
           and [num_layers, 1, seq_len, intermediate_size] for prefill.
        weights: W2 (down_proj) weight from the reference model.
        mode: "decode" or "prefill" (unused, kept for API consistency).

    Returns:
        w2_out tensor with shape [num_layers, 1, batch_size, hidden_size] for decode
        or [num_layers, 1, seq_len, hidden_size] for prefill.
    """
    w2_out = torch.nn.functional.linear(x, weights.w2)
    return w2_out


def ds_fused_ff2_ttnn(
    x: ttnn.Tensor,
    cfg: dict,
    mode: Literal["decode", "prefill"],
    seq_len: int,
    output_mem_config: ttnn.MemoryConfig | None = None,
    dram_interleaved_weight: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """
    TTNN implementation for FF2 fused op (down projection).

    This performs the down projection: w2(activated)

    Args:
        x: Input tensor (activated) from the mul operation.
        cfg: Configuration dictionary containing w2 config.
        mode: "decode" or "prefill" mode.
        seq_len: Sequence length (used for prefill program config).
        output_mem_config: Optional override for output memory config (useful for trace mode).
        dram_interleaved_weight: Optional DRAM-interleaved weight tensor for unit testing.
            When provided, this bypasses the DRAM-sharded weight config and uses standard matmul.
            This is needed because the production config uses MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
            which requires L1-sharded inputs, but unit tests use DRAM-interleaved inputs.

    Returns:
        w2_out tensor after down projection.
    """
    w2_cfg = dict(cfg["w2"])

    if dram_interleaved_weight is not None:
        # Unit test mode: use DRAM interleaved weight instead of DRAM-sharded weight
        # This allows testing with DRAM interleaved inputs (standard matmul)
        w2_cfg["input_tensor_b"] = dram_interleaved_weight
        w2_cfg["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        w2_cfg.pop("program_config", None)  # Use default matmul, not DRAM-sharded config
    elif output_mem_config is not None:
        w2_cfg["memory_config"] = output_mem_config

    if mode == "prefill":
        pc = MLP._get_prefill_pc(seq_len=seq_len, is_w2=True, **cfg["linear_pc_gen"])
        w2_out = ttnn.linear(x, program_config=pc, **w2_cfg)
    else:
        w2_out = ttnn.linear(x, **w2_cfg)

    return w2_out


def _compare_with_reference(
    tt_output: torch.Tensor, ref_output: torch.Tensor, expected_pcc: float, atol: float, rtol: float
) -> tuple[float, float]:
    """Compare TTNN output with reference and return metrics.

    Returns:
        Tuple of (pcc_value, max_abs_error) for logging to superset.
    """
    passing, pcc = comp_pcc(ref_output, tt_output, expected_pcc)
    max_abs_error = (tt_output - ref_output).abs().max().item()
    logger.info(f"PCC: {pcc}")
    logger.info(f"Max absolute error: {max_abs_error}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"
    # Note: For quantized weights (bfloat4_b), PCC is the primary metric.
    # torch.testing.assert_close may be too strict, so we only use it for sanity checks.
    try:
        torch.testing.assert_close(tt_output, ref_output, rtol=rtol, atol=atol)
    except AssertionError as e:
        logger.warning(f"assert_close failed but PCC passed: {e}")
    return pcc, max_abs_error


def _log_run_mode(mode: str, trace_mode: bool, program_cache_enabled: bool, seq_len: int, use_real_weights: bool):
    """Log the test run configuration."""
    logger.info("=== TEST RUN CONFIGURATION ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Sequence length: {seq_len}")
    logger.info(f"Trace mode: {trace_mode}")
    logger.info(f"Program cache enabled: {program_cache_enabled}")
    logger.info(f"Use real weights: {use_real_weights}")
    logger.info("===============================")


def _deallocate_outputs(outputs):
    """Deallocate outputs which can be a single tensor or tuple of tensors."""
    if isinstance(outputs, tuple):
        for out in outputs:
            ttnn.deallocate(out)
    else:
        ttnn.deallocate(outputs)


def _measure_perf_us(
    mesh_device: ttnn.MeshDevice, op_fn, warmup_iters: int, measure_iters: int, trace_mode: bool = False
) -> float:
    """
    Measure performance in microseconds.

    For trace mode, uses persistent output buffers to avoid host↔device IO during trace capture.
    """
    ttnn.synchronize_device(mesh_device)

    if trace_mode:
        # Trace mode: warm up in eager mode first, then capture and execute trace.
        # Warmup in eager mode to compile programs before trace capture
        for _ in range(warmup_iters):
            outputs = op_fn()
            ttnn.synchronize_device(mesh_device)
            _deallocate_outputs(outputs)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = op_fn()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        # Additional warmup with trace execution
        for _ in range(warmup_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)

        # Measure
        profiler.clear()
        profiler.start("ds_fused_ff2_perf")
        for _ in range(measure_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
        profiler.end("ds_fused_ff2_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        _deallocate_outputs(traced_output)
        return profiler.get("ds_fused_ff2_perf") * 1e6

    # Eager mode
    for _ in range(warmup_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        _deallocate_outputs(outputs)

    profiler.clear()
    profiler.start("ds_fused_ff2_perf")
    for _ in range(measure_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        _deallocate_outputs(outputs)
    profiler.end("ds_fused_ff2_perf", PERF_CNT=measure_iters)
    return profiler.get("ds_fused_ff2_perf") * 1e6


def _run_ds_fused_ff2_test(
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
    dram_interleaved_weight: ttnn.Tensor,
    use_real_weights: bool,
):
    # Log run configuration for superset
    _log_run_mode(mode, trace_mode, program_cache_enabled, seq_len, use_real_weights)

    # Log config for verification (Step 9 of AGENTS_GUIDE)
    logger.info("=== FF2 OP CONFIG VERIFICATION ===")
    logger.info(f"Input shape: {tt_input.shape}")
    logger.info(f"Input memory_config: {tt_input.memory_config()}")
    logger.info(f"W2 config keys: {run_config['w2'].keys()}")
    logger.info(f"W2 input_tensor_b shape: {run_config['w2']['input_tensor_b'].shape}")
    logger.info(f"W2 memory_config: {run_config['w2']['memory_config']}")
    logger.info(f"DRAM-interleaved weight shape: {dram_interleaved_weight.shape}")
    logger.info("=== END CONFIG VERIFICATION ===")

    # Run the fused op
    # Use dram_interleaved_weight for unit testing with DRAM interleaved inputs.
    # In the actual MLP, inputs come from the mul op which is L1 WIDTH_SHARDED
    # and uses the DRAM-sharded weights from cfg["w2"].
    tt_w2_out = ds_fused_ff2_ttnn(tt_input, run_config, mode, seq_len, dram_interleaved_weight=dram_interleaved_weight)

    # Convert to torch for comparison
    # For ff2 (down projection), the input is width-sharded on intermediate_size.
    # Each column device computes a PARTIAL product that needs to be SUMMED (reduce),
    # not concatenated. The output hidden_size is the same on all column devices.
    #
    # Mesh shape is (rows, cols) = (num_layers, 8)
    # - Rows (dim 0): different layers → concat
    # - Cols (dim -1): partial products → sum
    mesh_rows, mesh_cols = mesh_device.shape

    # Get per-device tensors and convert to torch
    device_tensors = ttnn.get_device_tensors(tt_w2_out)
    torch_tensors = [ttnn.to_torch(t, mesh_composer=None) for t in device_tensors]

    # torch_tensors is a flat list of 32 tensors (4 rows × 8 cols)
    # Reshape into [rows][cols] and sum across columns, concat across rows
    layer_outputs = []
    for row_idx in range(mesh_rows):
        # Sum partial products from all columns for this layer
        col_tensors = [torch_tensors[row_idx * mesh_cols + col_idx] for col_idx in range(mesh_cols)]
        layer_sum = col_tensors[0].clone()
        for t in col_tensors[1:]:
            layer_sum = layer_sum + t
        layer_outputs.append(layer_sum)

    # Stack layers on dim 0
    tt_w2_out_torch = torch.cat(layer_outputs, dim=0)

    logger.info("Comparing w2_out (down projection):")
    pcc_value, max_abs_error = _compare_with_reference(
        tt_w2_out_torch, ref_output, expected_pcc, expected_atol, expected_rtol
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = _get_int_env("DS_FF2_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = _get_int_env("DS_FF2_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn():
            return ds_fused_ff2_ttnn(
                tt_input, run_config, mode, seq_len, dram_interleaved_weight=dram_interleaved_weight
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
                "module": "mlp",
                "mesh_device": os.getenv("MESH_DEVICE", "TG"),
                "op_type": "ff2",
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
            return ds_fused_ff2_ttnn(tt_input, run_config, mode, seq_len)

        for _ in range(PERF_WARMUP_ITERS):
            outputs = op_fn()
            ttnn.synchronize_device(mesh_device)
            _deallocate_outputs(outputs)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            # Trace mode: capture trace and avoid any IO during trace execution
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
            _deallocate_outputs(traced_output)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                outputs = op_fn()
                ttnn.synchronize_device(mesh_device)
                _deallocate_outputs(outputs)
            signpost("stop")


def _build_ff2_weights(hf_config, use_real_weights: bool) -> FusedWeights:
    if use_real_weights:
        ref_model = DeepseekV3MLP(hf_config).eval().to(torch.bfloat16)
        state_dict = ref_model.state_dict()
    else:
        hidden = hf_config.hidden_size
        intermediate = hf_config.intermediate_size
        state_dict = {
            "gate_proj.weight": torch.randn(intermediate, hidden, dtype=torch.bfloat16),
            "up_proj.weight": torch.randn(intermediate, hidden, dtype=torch.bfloat16),
            "down_proj.weight": torch.randn(hidden, intermediate, dtype=torch.bfloat16),
        }
    return FusedWeights(w2=state_dict["down_proj.weight"])


def _build_ff2_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    use_real_weights: bool,
    mode: str,
    seq_len: int,
):
    """
    Build inputs for FF2 test.

    The FF2 operation (down projection) takes the output of the mul operation as input.
    The input tensor `activated` has shape:
    - Decode: [num_layers, 1, batch_size, intermediate_size] - full tensor before sharding
    - Prefill: [num_layers, 1, seq_len, intermediate_size] - full tensor before sharding

    The input is sharded across the mesh on dimension (0, -1) meaning:
    - First dimension (layers) is split across mesh rows
    - Last dimension (intermediate_size) is split across mesh columns

    For decode mode with batch_size=32, per device shape is:
    - [1, 1, 32, intermediate_size/mesh_width] = [1, 1, 32, 2304] for mesh_width=8
    """
    weights = _build_ff2_weights(hf_config, use_real_weights)

    state_dict = {
        "gate_proj.weight": torch.randn(hf_config.intermediate_size, hf_config.hidden_size, dtype=torch.bfloat16),
        "up_proj.weight": torch.randn(hf_config.intermediate_size, hf_config.hidden_size, dtype=torch.bfloat16),
        "down_proj.weight": weights.w2,
    }

    weight_config = get_test_weight_config(
        MLP,
        hf_config,
        (state_dict,) * mesh_device.shape[0],
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
    _, mesh_width = mesh_device.shape

    # The input to w2 is the output of the mul operation (activated)
    # This tensor is width-sharded with per_device_intermediate = intermediate_size / mesh_width
    # Full shape: [num_layers, 1, batch_size, intermediate_size]
    intermediate_size = hf_config.intermediate_size

    if mode == "decode":
        # Input shape for decode: [num_layers, 1, batch_size, intermediate_size]
        torch_input = torch.randn(num_layers, 1, batch_size, intermediate_size, dtype=torch.bfloat16)
    else:
        # Input shape for prefill: [num_layers, 1, seq_len, intermediate_size]
        torch_input = torch.randn(num_layers, 1, seq_len, intermediate_size, dtype=torch.bfloat16)

    # Convert to TTNN tensor - sharded across mesh on dims (0, -1)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Production w2 weight shape per device: [1, 2304, 7168] (3D)
    # Full shape before sharding: [num_layers, intermediate_size, hidden_size]
    # - dim 0 (num_layers) sharded across mesh rows
    # - dim 1 (intermediate_size) sharded across mesh cols
    # Per device: [1, 2304, 7168]
    w2_weight_transposed = weights.w2.t()  # [intermediate_size, hidden_size]
    w2_weight_3d = w2_weight_transposed.reshape(1, intermediate_size, hf_config.hidden_size)
    w2_weight_3d = w2_weight_3d.expand(num_layers, intermediate_size, hf_config.hidden_size).contiguous()
    tt_w2_weight_interleaved = ttnn.from_torch(
        w2_weight_3d,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat4_b,  # Match production weight dtype
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # For prefill, handle chunking if needed
    effective_seq_len = seq_len
    ref_input = torch_input
    num_layers_per_device = tt_input.shape[0]

    if mode == "prefill" and seq_len > run_config["max_rows"]:
        num_chunks = math.ceil(seq_len / run_config["max_rows"])
        tt_input = ttnn.reshape(tt_input, [num_layers_per_device, num_chunks, run_config["max_rows"], -1])
        # Reference uses full num_layers since we'll concat all device outputs
        ref_input = torch_input.reshape(num_layers, num_chunks, run_config["max_rows"], -1)
        effective_seq_len = run_config["max_rows"]

    # Compute reference output
    ref_output = ds_fused_ff2_reference(ref_input, weights, mode)

    return run_config, tt_input, ref_output, batch_size, effective_seq_len, tt_w2_weight_interleaved


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


def _skip_single_device_sharded():
    pytest.skip(
        "Single-device test is not applicable because ds_fused_ff2 relies on width-sharded matmuls across the mesh."
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # PCC ~0.97 is acceptable for bfloat4_b quantized weights
        ("decode", 1, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 128, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 1024, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 8192, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 131072, 0.97, 0.5, 0.5, 0.0),
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
def test_ds_fused_ff2(
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
):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"
        _maybe_skip_long_seq(seq_len)

    # Trace capture replays pre-compiled binaries. When program cache is disabled, ops may
    # trigger compilation/program writes during capture, which is forbidden and can TT_FATAL.
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    run_config, tt_input, ref_output, batch_size, effective_seq_len, tt_w2_weight = _build_ff2_inputs(
        mesh_device,
        hf_config,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        use_real_weights,
        mode,
        seq_len,
    )
    _run_ds_fused_ff2_test(
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
        effective_seq_len,
        batch_size,
        f"ds_fused_ff2_{mode}_seq{seq_len}",
        tt_w2_weight,
        use_real_weights,
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # PCC ~0.97 is acceptable for bfloat4_b quantized weights
        ("decode", 1, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 128, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 1024, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 8192, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 131072, 0.97, 0.5, 0.5, 0.0),
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
def test_ds_fused_ff2_single_device(
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
):
    _skip_single_device_sharded()


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
def test_ds_fused_ff2_device_perf(mode, seq_len):
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

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_fused_ff2_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/ff2/test_ds_fused_ff2.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_ds_fused_ff2 -k "{expr}"'

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
        ("prefill", 8192),
        ("prefill", 131072),
    ],
)
def test_ds_fused_ff2_single_device_device_perf(mode, seq_len):
    _skip_single_device_sharded()


if __name__ == "__main__":
    pytest.main([__file__])
