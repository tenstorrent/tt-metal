# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import (
    collect_device_perf,
    compare_with_reference,
    deallocate_outputs,
    get_int_env,
    log_run_mode,
    maybe_skip_long_seq,
    measure_perf_us,
    skip_single_device_sharded,
)
from models.demos.deepseek_v3.tt.mlp.mlp import MLP
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_FF2_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 234.146, "op_to_op": 1809.236},  # Measured: kernel=212.86, op_to_op=1644.76
    ("prefill", 128): {"kernel": 585.915, "op_to_op": 32558.130},  # Measured: kernel=532.65, op_to_op=29598.30
}


@dataclass
class FusedWeights:
    w2: torch.Tensor  # down_proj weight


def ds_ff2_reference(x: torch.Tensor, weights: FusedWeights, mode: Literal["decode", "prefill"]) -> torch.Tensor:
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


def ds_ff2_ttnn(
    x: ttnn.Tensor,
    cfg: dict,
    mode: Literal["decode", "prefill"],
    seq_len: int,
    output_mem_config: ttnn.MemoryConfig | None = None,
    dram_interleaved_weight: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """TTNN implementation for FF2 op (down projection).

    Note: output_mem_config and dram_interleaved_weight kept for backward compatibility but ignored.
    """
    # Compute program config for prefill if needed
    program_config = None
    if mode == "prefill":
        program_config = MLP._get_prefill_pc(seq_len=seq_len, is_w2=True, **cfg["linear_pc_gen"])

    return MLP._fwd_ff2(x, cfg["w2"], program_config=program_config)


def _run_ds_ff2_test(
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
    log_run_mode(mode, trace_mode, program_cache_enabled, seq_len, use_real_weights=use_real_weights)

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
    tt_w2_out = ds_ff2_ttnn(tt_input, run_config, mode, seq_len, dram_interleaved_weight=dram_interleaved_weight)

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
    pcc_value, max_abs_error = compare_with_reference(
        tt_w2_out_torch, ref_output, expected_pcc, expected_atol, expected_rtol, strict_assert=False
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = get_int_env("DS_FF2_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = get_int_env("DS_FF2_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn():
            return ds_ff2_ttnn(tt_input, run_config, mode, seq_len, dram_interleaved_weight=dram_interleaved_weight)

        perf_us = measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
            profiler_name="ds_ff2_perf",
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
            return ds_ff2_ttnn(tt_input, run_config, mode, seq_len)

        for _ in range(PERF_WARMUP_ITERS):
            outputs = op_fn()
            ttnn.synchronize_device(mesh_device)
            deallocate_outputs(outputs)

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
            deallocate_outputs(traced_output)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                outputs = op_fn()
                ttnn.synchronize_device(mesh_device)
                deallocate_outputs(outputs)
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

    batch_size = USERS_PER_ROW  # Always 32 for all modes
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
    # Match production: mul output is L1 WIDTH_SHARDED for decode (cfg["mul"]["memory_config"])
    # For prefill, use DRAM INTERLEAVED (production uses DRAM for prefill)
    if mode == "decode":
        from models.demos.deepseek_v3.utils.config_helpers import get_activation_sharding_core_counts_for_dram_matmul

        max_num_cores = mesh_device.core_grid.x * mesh_device.core_grid.y
        inner_num_cores = max(
            get_activation_sharding_core_counts_for_dram_matmul(intermediate_size // mesh_width, max_num_cores)
        )

        # Create L1 WIDTH_SHARDED memory config matching production mul output
        activation_mem_config = ttnn.create_sharded_memory_config_(
            shape=(
                32,  # USERS_PER_ROW (batch_size for decode)
                intermediate_size // mesh_width // inner_num_cores,
            ),
            core_grid=ttnn.num_cores_to_corerangeset(
                inner_num_cores,
                ttnn.CoreCoord(mesh_device.core_grid.x, mesh_device.core_grid.y),
                row_wise=True,
            ),
            strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        # Prefill uses DRAM_MEMORY_CONFIG (cfg["mul"]["memory_config"] = DRAM for prefill)
        activation_mem_config = ttnn.DRAM_MEMORY_CONFIG

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=activation_mem_config,
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
    ref_output = ds_ff2_reference(ref_input, weights, mode)

    return run_config, tt_input, ref_output, batch_size, effective_seq_len, tt_w2_weight_interleaved


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # PCC ~0.97 is acceptable for bfloat4_b quantized weights
        # batch_size=32 for all modes
        ("decode", 1, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 128, 0.97, 0.5, 0.5, 0.0),
        pytest.param(
            "prefill",
            1024,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            8192,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            32768,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            131072,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
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
def test_ds_ff2(
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
    _run_ds_ff2_test(
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
        f"ds_ff2_{mode}_seq{seq_len}",
        tt_w2_weight,
        use_real_weights,
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # PCC ~0.97 is acceptable for bfloat4_b quantized weights
        # batch_size=32 for all modes
        ("decode", 1, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 128, 0.97, 0.5, 0.5, 0.0),
        pytest.param(
            "prefill",
            1024,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            8192,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            32768,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            131072,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
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
def test_ds_ff2_single_device(
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
    skip_single_device_sharded("ds_ff2")


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        pytest.param("prefill", 1024, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 8192, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 32768, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 131072, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
    ],
)
def test_ds_ff2_device_perf(mode, seq_len):
    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"
        maybe_skip_long_seq(seq_len, LONG_SEQ_ENV_VAR)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0]

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_ff2_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_ds_ff2 -k "{expr}"'

    perf_profiler.start("run")
    perf_profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = collect_device_perf(
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
        pytest.param("prefill", 1024, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 8192, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 32768, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 131072, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
    ],
)
def test_ds_ff2_single_device_device_perf(mode, seq_len):
    skip_single_device_sharded("ds_ff2")


if __name__ == "__main__":
    pytest.main([__file__])
