# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for RMSNorm (non-distributed).

This tests the single-op RMSNorm used for smaller dimensions in DeepSeek:
- kv_lora_rank (512) - for kv_a_layernorm in MLA
- q_lora_rank (1536) - for q_a_layernorm in MLA

Unlike DistributedRMSNorm, this version does NOT shard the hidden dimension
across devices, so no all-gather is needed.
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
from models.demos.deepseek_v3.tt.rms_norm.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

LONG_SEQ_ENV_VAR = "DEEPSEEK_V3_LONG_SEQ_TESTS"
DEVICE_PERF_ENV_VAR = "DS_RMS_NORM_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 30.492, "op_to_op": 754.083},  # Measured: kernel=27.72, op_to_op=685.53
    ("prefill", 128): {"kernel": 64.911, "op_to_op": 15608.395},  # Measured: kernel=59.01, op_to_op=14189.45
    ("prefill", 1024, "kv_lora_rank"): {
        "kernel": 45.244,
        "op_to_op": 37423.566,
    },  # Measured: kernel=41.131, op_to_op=34021.424
    ("prefill", 1024, "q_lora_rank"): {
        "kernel": 111.938,
        "op_to_op": 106094.376,
    },  # Measured: kernel=101.76, op_to_op=96449.43
}


def ds_rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Reference implementation for RMSNorm (non-distributed).

    This is standard RMSNorm without any tensor parallelism.

    Args:
        x: Input tensor of shape [num_layers, batch_size, seq_len, hidden_size]
           where hidden_size is kv_lora_rank (512) or q_lora_rank (1536)
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


def ds_rms_norm_ttnn(
    x: ttnn.Tensor,
    cfg: dict,
) -> ttnn.Tensor:
    """
    TTNN implementation for RMSNorm (non-distributed).

    This is a single ttnn.rms_norm op, no all-gather needed since the
    hidden dimension is NOT sharded across devices.

    Args:
        x: Input tensor (INTERLEAVED DRAM)
        cfg: Configuration dictionary containing epsilon, weight, compute_kernel_config

    Returns:
        Normalized output tensor (same shape as input)
    """
    program_config = RMSNorm._get_pc(x.memory_config())
    return RMSNorm._fwd_rms_norm(x, cfg, program_config)


def _run_ds_rms_norm_test(
    mesh_device: ttnn.MeshDevice,
    run_config: dict,
    tt_input: ttnn.Tensor,
    ref_output: torch.Tensor,
    hidden_size: int,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    mode: str,
    seq_len: int,
    batch_size: int,
    hf_config_size_attr: str,
    step_prefix: str,
):
    # Log run configuration for superset
    log_run_mode(mode, trace_mode, program_cache_enabled, seq_len, hf_config_size_attr=hf_config_size_attr)

    tt_output = ds_rms_norm_ttnn(tt_input, run_config)

    # Convert output back to torch - concat across mesh rows (dim 0 only, not dim -1)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -1)),
    )
    # Slice to actual hidden_size (output may be padded to tile width)
    tt_output_torch = tt_output_torch[..., :hidden_size]

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

        warmup_iters = get_int_env("DS_RMS_NORM_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = get_int_env("DS_RMS_NORM_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn():
            return ds_rms_norm_ttnn(tt_input, run_config)

        perf_us = measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
            profiler_name="ds_rms_norm_perf",
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
                "op_type": "rms_norm",
                "hf_config_size_attr": hf_config_size_attr,
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
            return ds_rms_norm_ttnn(tt_input, run_config)

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


def _build_rms_norm_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    force_recalculate_weight_config: bool,
    use_real_weights: bool,
    mode: str,
    seq_len: int,
    hf_config_size_attr: str,
    state_dict: dict[str, torch.Tensor] | None,
):
    """Build inputs for RMSNorm test.

    Args:
        mesh_device: The mesh device
        hf_config: HuggingFace config
        cache_path: Path for weight caching
        force_recalculate_weight_config: Whether to force recalculate weights
        use_real_weights: Whether to use real model weights
        mode: "decode" or "prefill"
        seq_len: Sequence length (batch size for decode mode)
        hf_config_size_attr: "kv_lora_rank" (512) or "q_lora_rank" (1536)
        state_dict: Model state dict (needed if use_real_weights=True)
    """
    num_module_layers = mesh_device.shape[0]

    # Get hidden_size based on hf_config_size_attr
    hidden_size = getattr(hf_config, hf_config_size_attr)
    epsilon = hf_config.rms_norm_eps

    # Get reference weights
    reference_model = DeepseekV3RMSNorm(
        hidden_size=hidden_size,
        eps=epsilon,
    ).eval()

    # Determine the reference path based on hf_config_size_attr
    if hf_config_size_attr == "kv_lora_rank":
        reference_layernorm_path = "model.layers.0.self_attn.kv_a_layernorm"
    else:  # q_lora_rank
        reference_layernorm_path = "model.layers.0.self_attn.q_a_layernorm"

    if use_real_weights and state_dict is not None:
        # Use real weights from the model
        norm_state_dict = sub_state_dict(state_dict, reference_layernorm_path + ".")
        reference_model.load_state_dict({k: v.to(torch.float32) for k, v in norm_state_dict.items()})
        norm_state_dict = {k: v.to(torch.bfloat16) for k, v in norm_state_dict.items()}
    else:
        norm_state_dict = reference_model.to(torch.bfloat16).state_dict()

    weight = norm_state_dict["weight"]

    # Generate TTNN configs
    weight_config = get_test_weight_config(
        RMSNorm,
        hf_config,
        [norm_state_dict] * num_module_layers,
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(RMSNorm, mode, hf_config, mesh_device)
    model_state = RMSNorm.create_state(hf_config, mesh_device)  # No CCL needed for non-distributed
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

    # Convert to TTNN with INTERLEAVED DRAM memory config
    # Shard only across mesh rows (dim 0 for layers), NOT across mesh cols (dim -1)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Compute reference output using PyTorch
    ref_output = ds_rms_norm_reference(torch_input, weight, epsilon)

    return run_config, tt_input, ref_output, batch_size, hidden_size


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # PCC ~0.98 is typical for RMSNorm
        # Decode with seq_len=1, batch_size=32 (USERS_PER_ROW)
        ("decode", 1, 0.98, 0.5, 0.5, 0.0),  # TODO: set real perf targets
        ("prefill", 128, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 1024, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 8192, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 32768, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 131072, 0.98, 0.5, 0.5, 0.0),  # 128k
    ],
)
@pytest.mark.parametrize(
    "hf_config_size_attr",
    ["kv_lora_rank", "q_lora_rank"],
    ids=["kv_lora_rank_512", "q_lora_rank_1536"],
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
def test_ds_rms_norm(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    hf_config_size_attr,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    hf_config,
    cache_path,
    mesh_device,
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

    run_config, tt_input, ref_output, batch_size, hidden_size = _build_rms_norm_inputs(
        mesh_device,
        hf_config,
        cache_path,
        force_recalculate_weight_config,
        use_real_weights,
        mode,
        seq_len,
        hf_config_size_attr,
        state_dict if use_real_weights else None,
    )
    _run_ds_rms_norm_test(
        mesh_device,
        run_config,
        tt_input,
        ref_output,
        hidden_size,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        mode,
        seq_len,
        batch_size,
        hf_config_size_attr,
        f"ds_rms_norm_{mode}_seq{seq_len}_{hf_config_size_attr}",
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 128, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 1024, 0.98, 0.5, 0.5, 0.0),
        ("prefill", 131072, 0.98, 0.5, 0.5, 0.0),
    ],
)
@pytest.mark.parametrize(
    "hf_config_size_attr",
    ["kv_lora_rank", "q_lora_rank"],
    ids=["kv_lora_rank_512", "q_lora_rank_1536"],
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
def test_ds_rms_norm_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    hf_config_size_attr,
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
    """Single device test for RMSNorm.

    Unlike DistributedRMSNorm, the non-distributed RMSNorm does NOT have CCL ops,
    so single device test is applicable. However, since this test uses the same
    code path and the op runs independently on each device (no sharding on hidden dim),
    the multi-device test already exercises the single-device behavior.
    """
    # For now, skip with a note that the multi-device test already covers this
    pytest.skip(
        "Single-device test skipped: RMSNorm runs identically on each device "
        "(hidden dim not sharded). Multi-device test already validates correctness."
    )


@pytest.mark.parametrize(
    "mode, seq_len, hf_config_size_attr",
    [
        ("decode", 1, "kv_lora_rank"),
        ("decode", 1, "q_lora_rank"),
        ("prefill", 128, "kv_lora_rank"),
        ("prefill", 128, "q_lora_rank"),
        pytest.param(
            "prefill", 1024, "kv_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 1024, "q_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 8192, "kv_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 8192, "q_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 32768, "kv_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 32768, "q_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 131072, "kv_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 131072, "q_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
    ],
)
def test_ds_rms_norm_device_perf(mode, seq_len, hf_config_size_attr):
    maybe_skip_long_seq(seq_len, LONG_SEQ_ENV_VAR)

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    # batch_size=32 (USERS_PER_ROW) for all modes
    batch_size = USERS_PER_ROW

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_rms_norm_device_perf_{mode}_seq{seq_len}_{hf_config_size_attr}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    size_filter = "kv_lora_rank_512" if hf_config_size_attr == "kv_lora_rank" else "q_lora_rank_1536"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and real_weights and {size_filter}"
    command = f'pytest {test_path}::test_ds_rms_norm -k "{expr}"'

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
    targets = DEVICE_PERF_TARGETS_US.get((mode, seq_len, hf_config_size_attr))
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
    "mode, seq_len, hf_config_size_attr",
    [
        ("decode", 1, "kv_lora_rank"),
        ("decode", 1, "q_lora_rank"),
        ("prefill", 128, "kv_lora_rank"),
        ("prefill", 128, "q_lora_rank"),
        pytest.param(
            "prefill", 1024, "kv_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 1024, "q_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 8192, "kv_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 8192, "q_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 32768, "kv_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
        pytest.param(
            "prefill", 32768, "q_lora_rank", marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")
        ),
    ],
)
def test_ds_rms_norm_single_device_device_perf(mode, seq_len, hf_config_size_attr):
    pytest.skip(
        "Single-device device perf test skipped: RMSNorm runs identically on each device "
        "(hidden dim not sharded). Multi-device test already validates performance."
    )


if __name__ == "__main__":
    pytest.main([__file__])
