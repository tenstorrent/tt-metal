# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for Distributed RMSNorm.

This tests the distributed RMSNorm sequence:
1. ttnn.rms_norm_pre_all_gather - compute local statistics
2. ttnn.experimental.all_gather_async - gather stats across devices
3. ttnn.rms_norm_post_all_gather - apply normalization with gathered stats
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3RMSNorm
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import (
    collect_device_perf,
    compare_with_reference,
    get_int_env,
    log_run_mode,
    maybe_skip_long_seq,
    measure_perf_us,
)
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_DISTRIBUTED_NORM_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 48.851, "op_to_op": 999.966},  # Measured: kernel=44.41, op_to_op=909.06
    ("prefill", 128): {
        "kernel": 262.031,
        "op_to_op": 257954.774,
    },  # Measured: kernel=239.33, op_to_op=234504.34 (with margin)
}


def ds_distributed_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Reference implementation for Distributed RMSNorm.

    In the reference model (without tensor parallelism), this is standard RMSNorm.
    The distributed version gathers statistics across devices, but the math is the same.

    Args:
        x: Input tensor of shape [num_layers, batch_size, seq_len, hidden_size]
        weight: RMSNorm weight tensor of shape [hidden_size]
        epsilon: Small constant for numerical stability

    Returns:
        Normalized output tensor of same shape as input
    """
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    return (weight.to(torch.float32) * x).to(input_dtype)


def ds_distributed_norm_ttnn(
    x: ttnn.Tensor,
    cfg: dict,
    ccl,
) -> ttnn.Tensor:
    """
    TTNN implementation for Distributed RMSNorm.

    This performs the distributed RMSNorm in three steps:
    1. Compute local statistics (partial sum of squares)
    2. AllGather statistics across devices
    3. Apply normalization using gathered statistics

    Args:
        x: Input tensor sharded across devices (WIDTH_SHARDED)
        cfg: Configuration dictionary containing all op configs
        ccl: CCL runtime object for all-gather

    Returns:
        Normalized output tensor (same shape and sharding as input)
    """
    program_config = DistributedRMSNorm._get_pc(x.memory_config())

    # Step 1: Compute local statistics
    tt_stats = DistributedRMSNorm._fwd_rms_norm_pre_all_gather(x, cfg, program_config)

    # Step 2: AllGather stats across devices
    tt_gathered_stats = DistributedRMSNorm._fwd_all_gather_stats(tt_stats, cfg, ccl)
    ttnn.deallocate(tt_stats)

    # Step 3: Apply normalization with gathered stats
    tt_out = DistributedRMSNorm._fwd_rms_norm_post_all_gather(x, tt_gathered_stats, cfg, program_config)
    ttnn.deallocate(tt_gathered_stats)

    return tt_out


def _run_ds_distributed_norm_test(
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
    log_run_mode(mode, trace_mode, program_cache_enabled, seq_len)

    tt_output = ds_distributed_norm_ttnn(tt_input, run_config, ccl)

    # Convert output back to torch - concat across mesh
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -1)),
    )

    pcc_value, max_abs_error = compare_with_reference(
        tt_output_torch,
        ref_output,
        expected_pcc,
        expected_atol,
        expected_rtol,
        convert_to_float=True,
        strict_assert=False,
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = get_int_env("DS_DISTRIBUTED_NORM_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = get_int_env("DS_DISTRIBUTED_NORM_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn():
            return ds_distributed_norm_ttnn(tt_input, run_config, ccl)

        perf_us = measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
            profiler_name="ds_distributed_norm_perf",
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
                "module": "rms_norm",
                "mesh_device": os.getenv("MESH_DEVICE", "TG"),
                "op_type": "distributed_norm",
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
            return ds_distributed_norm_ttnn(tt_input, run_config, ccl)

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


def _build_distributed_norm_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    use_real_weights: bool,
    mode: str,
    seq_len: int,
    state_dict: dict[str, torch.Tensor] | None,
    reference_layernorm_path: str | None,
):
    """Build inputs for distributed norm test.

    Args:
        mesh_device: The mesh device
        hf_config: HuggingFace config
        cache_path: Path for weight caching
        ccl: CCL object for all-gather
        force_recalculate_weight_config: Whether to force recalculate weights
        use_real_weights: Whether to use real model weights
        mode: "decode" or "prefill"
        seq_len: Sequence length (batch size for decode mode)
        state_dict: Model state dict (needed if use_real_weights=True)
        reference_layernorm_path: Path to layernorm in state dict
    """
    num_module_layers = mesh_device.shape[0]
    hidden_size = hf_config.hidden_size
    epsilon = hf_config.rms_norm_eps

    # Get reference weights
    reference_model = DeepseekV3RMSNorm(
        hidden_size=hidden_size,
        eps=epsilon,
    ).eval()

    if use_real_weights and state_dict is not None and reference_layernorm_path is not None:
        # Use real weights from the model
        norm_state_dict = sub_state_dict(state_dict, reference_layernorm_path + ".")
        reference_model.load_state_dict({k: v.to(torch.float32) for k, v in norm_state_dict.items()})
        norm_state_dict = {k: v.to(torch.bfloat16) for k, v in norm_state_dict.items()}
    else:
        norm_state_dict = reference_model.to(torch.bfloat16).state_dict()

    weight = norm_state_dict["weight"]

    # Generate TTNN configs
    weight_config = get_test_weight_config(
        DistributedRMSNorm,
        hf_config,
        [norm_state_dict] * num_module_layers,
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(DistributedRMSNorm, mode, hf_config, mesh_device)
    model_state = DistributedRMSNorm.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Input shape: [num_layers, 1, height, hidden_size]
    # RMSNorm convention (from original test_rms_norm.py):
    # - Decode: height = 32 (USERS_PER_ROW users × 1 token each)
    # - Prefill: height = seq_len (1 user × seq_len tokens)
    if mode == "decode":
        batch_size = USERS_PER_ROW
        effective_height = batch_size * seq_len  # 32 × 1 = 32
    else:
        batch_size = 1
        effective_height = seq_len  # 1 × seq_len = seq_len
    torch_input = torch.randn(num_module_layers, 1, effective_height, hidden_size, dtype=torch.bfloat16)

    # Convert to TTNN with WIDTH_SHARDED memory config
    # Shard across mesh rows (dim 0) and mesh cols (dim -1 = hidden_size)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, -1)),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    # Compute reference output using PyTorch
    ref_output = ds_distributed_norm_reference(torch_input, weight, epsilon)

    return run_config, tt_input, ref_output, batch_size


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # For decode mode, seq_len=1 with batch_size=32 (USERS_PER_ROW)
        # PCC ~0.98 is typical for RMSNorm
        ("decode", 1, 0.98, 0.5, 0.5, 0.0),  # TODO: set real perf targets
        ("prefill", 128, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 1024, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 8192, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 32768, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 131072, 0.98, 0.5, 0.5, 0.0),  # 128k
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
            "trace_region_size": 2967552,
        }
    ],
    indirect=True,
)
def test_ds_distributed_norm(
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
    state_dict: dict[str, torch.Tensor],
    is_ci_env,
):
    # CI skip logic: keep only decode/1/trace and prefill/128/eager in CI with program_cache and real_weights
    if is_ci_env:
        ci_keep = (mode == "decode" and seq_len == 1 and trace_mode and program_cache_enabled and use_real_weights) or (
            mode == "prefill" and seq_len == 128 and not trace_mode and program_cache_enabled and use_real_weights
        )
        if not ci_keep:
            pytest.skip("CI test only runs decode/1/trace and prefill/128/eager with program_cache and real_weights")

    # Trace capture replays pre-compiled binaries. When program cache is disabled, ops may
    # trigger compilation/program writes during capture, which is forbidden and can TT_FATAL.
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # Use input_layernorm path for real weights
    reference_layernorm_path = "model.layers.0.input_layernorm" if use_real_weights else None

    run_config, tt_input, ref_output, batch_size = _build_distributed_norm_inputs(
        mesh_device,
        hf_config,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        use_real_weights,
        mode,
        seq_len,
        state_dict if use_real_weights else None,
        reference_layernorm_path,
    )
    _run_ds_distributed_norm_test(
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
        f"ds_distributed_norm_{mode}_seq{seq_len}",
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
def test_ds_distributed_norm_device_perf(mode, seq_len):
    maybe_skip_long_seq(seq_len, LONG_SEQ_ENV_VAR)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0]

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_distributed_norm_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and real_weights"
    command = f'pytest {test_path}::test_ds_distributed_norm -k "{expr}"'

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


if __name__ == "__main__":
    pytest.main([__file__])
