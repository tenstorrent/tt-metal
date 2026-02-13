# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import (
    collect_device_perf,
    compare_with_reference,
    get_int_env,
    log_run_mode,
    maybe_skip_long_seq,
    measure_perf_us,
)
from models.demos.deepseek_v3.tt.mlp.mlp import MLP
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, even_int_div
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_MUL_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 30.808, "op_to_op": 1145.261},  # Measured: kernel=28.01, op_to_op=1041.15
    ("prefill", 128): {"kernel": 66.708, "op_to_op": 43952.403},  # Measured: kernel=60.64, op_to_op=39956.73
}


def ds_mul_reference(
    w1_out: torch.Tensor,
    w3_out: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation for the mul fused op with SILU activation.

    The ttnn.mul operation includes a built-in SILU activation applied to input_tensor_a
    before multiplication.

    Args:
        w1_out: Input tensor that will have SILU applied (corresponds to w1_out_activated before fusing)
        w3_out: Input tensor to multiply with

    Returns:
        Output tensor: silu(w1_out) * w3_out
    """
    # Apply SILU activation to w1_out, then multiply with w3_out
    w1_activated = torch.nn.functional.silu(w1_out.float()).to(w1_out.dtype)
    return w1_activated * w3_out


def ds_mul_ttnn(
    w1_out: ttnn.Tensor,
    w3_out: ttnn.Tensor,
    cfg: dict,
    mode: str,
    output_mem_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    TTNN implementation for the mul fused op with SILU activation.

    This performs: silu(w1_out) * w3_out
    - For prefill: Uses fused activation in ttnn.mul
    - For decode: Applies MLP._silu_workaround before calling the wrapper

    Args:
        w1_out: Input tensor that will have SILU applied
        w3_out: Input tensor to multiply with
        cfg: Configuration dictionary containing mul config
        mode: "decode" or "prefill"
        output_mem_config: Optional override for output memory config (useful for unit tests)

    Returns:
        Output tensor after silu+mul
    """
    mul_cfg = dict(cfg["mul"])

    # Allow test to override memory_config if needed (e.g., when inputs are in DRAM)
    if output_mem_config is not None:
        mul_cfg["memory_config"] = output_mem_config
        # When using DRAM output, we need to apply SILU separately
        if output_mem_config == ttnn.DRAM_MEMORY_CONFIG:
            mul_cfg.pop("input_tensor_a_activations", None)
            # Apply SILU before mul for DRAM output
            if mode == "decode":
                w1_out = MLP._silu_workaround(w1_out)
            else:  # prefill
                w1_out = ttnn.silu(w1_out, memory_config=output_mem_config)
    else:
        # For non-DRAM output in decode mode, apply the workaround before calling the wrapper
        if mode == "decode":
            w1_out = MLP._silu_workaround(w1_out)

    # Call the MLP wrapper
    return MLP._fwd_mul(w1_out, w3_out, mul_cfg)


def _run_ds_mul_test(
    mesh_device: ttnn.MeshDevice,
    run_config: dict,
    tt_w1_out: ttnn.Tensor,
    tt_w3_out: ttnn.Tensor,
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
    # Log run configuration for superset
    log_run_mode(mode, trace_mode, program_cache_enabled, seq_len)

    # Log config for verification (Step 9 of AGENTS_GUIDE)
    logger.info(f"=== MUL OP CONFIG VERIFICATION ===")
    logger.info(f"Input w1_out shape: {tt_w1_out.shape}")
    logger.info(f"Input w3_out shape: {tt_w3_out.shape}")
    logger.info(f"Input w1_out memory_config: {tt_w1_out.memory_config()}")
    logger.info(f"Mul config from run_config: {run_config['mul']}")
    logger.info(f"=== END CONFIG VERIFICATION ===")

    # Use DRAM output since inputs are in DRAM (unit test simplification)
    # In actual MLP forward, inputs would be L1_WIDTH_SHARDED from linear ops
    tt_output = ds_mul_ttnn(tt_w1_out, tt_w3_out, run_config, mode, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)

    # Collect output from all devices and compare
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    pcc_value, max_abs_error = compare_with_reference(
        tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = get_int_env("DS_MUL_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = get_int_env("DS_MUL_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn():
            return ds_mul_ttnn(tt_w1_out, tt_w3_out, run_config, mode, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)

        perf_us = measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
            profiler_name="ds_mul_perf",
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
                "op_type": "mul",
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
            return ds_mul_ttnn(tt_w1_out, tt_w3_out, run_config, mode, output_mem_config=ttnn.DRAM_MEMORY_CONFIG)

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


def _build_mul_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    mode: str,
    seq_len: int,
):
    from models.demos.deepseek_v3.tt.mlp.mlp import MLP

    weight_config = get_test_weight_config(
        MLP,
        hf_config,
        (None,) * mesh_device.shape[0],  # No weights needed for mul op
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

    # Get dimensions from config
    hidden_dim = hf_config.intermediate_size
    _, mesh_width = mesh_device.shape

    # Input shape for decode: [num_layers, 1, batch_size, hidden_dim]
    # Input shape for prefill: [num_layers, 1, seq_len, hidden_dim]
    if mode == "decode":
        # Input tensors come from w1/w3 linear outputs in decode mode
        # Shape: [1, 1, batch_size, hidden_dim] per device before sharding
        torch_w1_out = torch.randn(num_layers, 1, batch_size, hidden_dim, dtype=torch.bfloat16)
        torch_w3_out = torch.randn(num_layers, 1, batch_size, hidden_dim, dtype=torch.bfloat16)
    else:
        torch_w1_out = torch.randn(num_layers, 1, seq_len, hidden_dim, dtype=torch.bfloat16)
        torch_w3_out = torch.randn(num_layers, 1, seq_len, hidden_dim, dtype=torch.bfloat16)

    # Compute reference output before sharding
    ref_output = ds_mul_reference(torch_w1_out, torch_w3_out)

    # Convert to TTNN tensors - always start in DRAM, then reshard if needed
    # This matches how the MLP module receives inputs (from previous ops in DRAM or already sharded)
    tt_w1_out = ttnn.from_torch(
        torch_w1_out,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_w3_out = ttnn.from_torch(
        torch_w3_out,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    return run_config, tt_w1_out, tt_w3_out, ref_output, batch_size


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # batch_size=32 for all modes
        ("decode", 1, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 1024, 0.9999, 0.2, 0.2, 0.0),
        pytest.param(
            "prefill",
            131072,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
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
def test_ds_mul(
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
        # Skip removed: now handled by pytest.param marks for seq_len > 8192

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    run_config, tt_w1_out, tt_w3_out, ref_output, batch_size = _build_mul_inputs(
        mesh_device,
        hf_config,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        mode,
        seq_len,
    )
    _run_ds_mul_test(
        mesh_device,
        run_config,
        tt_w1_out,
        tt_w3_out,
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
        f"ds_mul_{mode}_seq{seq_len}",
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # batch_size=32 for all modes
        ("decode", 1, 0.9999, 0.2, 0.2, 0.0),
        ("prefill", 128, 0.9999, 0.2, 0.2, 0.0),
        pytest.param(
            "prefill",
            1024,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            8192,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            32768,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
        pytest.param(
            "prefill",
            131072,
            0.9999,
            0.2,
            0.2,
            0.0,
            marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI"),
        ),
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
def test_ds_mul_single_device(
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
    """
    Single device test for the mul fused op.

    This test runs on a single device from the mesh. Since mul doesn't use CCL ops,
    we can run a proper single-device test.
    """
    # Trace capture replays pre-compiled binaries. When program cache is disabled, ops may
    # trigger compilation/program writes during capture, which is forbidden and can TT_FATAL.
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"
        # Skip removed: now handled by pytest.param marks for seq_len > 8192

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # For single device test, we need to extract single device shapes
    # The per-device shape is the full tensor shape divided by the mesh dims
    from models.demos.deepseek_v3.tt.mlp.mlp import MLP

    weight_config = get_test_weight_config(
        MLP,
        hf_config,
        (None,) * mesh_device.shape[0],
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

    # Get dimensions from config
    hidden_dim = hf_config.intermediate_size
    _, mesh_width = mesh_device.shape

    # Per-device hidden_dim (sharded on last dimension)
    per_device_hidden_dim = even_int_div(hidden_dim, mesh_width)

    # Single device input shape
    if mode == "decode":
        torch_w1_out = torch.randn(1, 1, batch_size, per_device_hidden_dim, dtype=torch.bfloat16)
        torch_w3_out = torch.randn(1, 1, batch_size, per_device_hidden_dim, dtype=torch.bfloat16)
    else:
        torch_w1_out = torch.randn(1, 1, seq_len, per_device_hidden_dim, dtype=torch.bfloat16)
        torch_w3_out = torch.randn(1, 1, seq_len, per_device_hidden_dim, dtype=torch.bfloat16)

    # Compute reference output
    ref_output = ds_mul_reference(torch_w1_out, torch_w3_out)

    # Get single device from mesh
    single_device = mesh_device.get_device(0)

    # Get the memory config for the mul op inputs
    if mode == "decode":
        input_memory_config = run_config["w1"]["memory_config"]
    else:
        input_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Convert to TTNN tensors on single device
    tt_w1_out = ttnn.from_torch(
        torch_w1_out,
        device=single_device,
        dtype=ttnn.bfloat16,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_w3_out = ttnn.from_torch(
        torch_w3_out,
        device=single_device,
        dtype=ttnn.bfloat16,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run the mul op
    tt_output = ttnn.mul(tt_w1_out, tt_w3_out, **run_config["mul"])

    # Convert output to torch
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare with reference
    compare_with_reference(tt_output_torch, ref_output, expected_pcc, expected_atol, expected_rtol)


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
def test_ds_mul_device_perf(mode, seq_len):
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
    step_name = f"ds_mul_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_ds_mul -k "{expr}"'

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
def test_ds_mul_single_device_device_perf(mode, seq_len):
    """
    Single device device performance test for the mul fused op.
    """
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
    step_name = f"ds_mul_single_device_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_ds_mul_single_device -k "{expr}"'

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
