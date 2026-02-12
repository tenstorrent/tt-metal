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
DEVICE_PERF_ENV_VAR = "DS_FF1_3_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 285.439, "op_to_op": 2110.702},  # Measured: kernel=259.49, op_to_op=1918.82
    ("prefill", 128): {"kernel": 37307.050, "op_to_op": 6490894.971},  # Measured: kernel=33915.50, op_to_op=5900813.61
}


@dataclass
class FusedWeights:
    w1: torch.Tensor
    w3: torch.Tensor


def ds_ff1_3_reference(
    x: torch.Tensor, weights: FusedWeights, mode: Literal["decode", "prefill"]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for FF1/3 fused op (gate + up projections only).

    Args:
        x: Input tensor shaped like the module input after all-gather. Shape is
           [num_layers, seq_len, batch, hidden] for decode and
           [num_layers, batch, seq_len, hidden] for prefill (chunked if needed).
        weights: W1 (gate_proj) and W3 (up_proj) weights from the reference model.
        mode: "decode" or "prefill" (unused, kept for API consistency).

    Returns:
        Tuple of (w1_out, w3_out) tensors.
    """
    w1_out = torch.nn.functional.linear(x, weights.w1)
    w3_out = torch.nn.functional.linear(x, weights.w3)
    return w1_out, w3_out


def ds_ff1_3_ttnn(
    x: ttnn.Tensor, cfg: dict, mode: Literal["decode", "prefill"], seq_len: int
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """TTNN implementation for FF1/3 op (gate + up projections)."""
    # Compute program config for prefill if needed
    program_config = None
    if mode == "prefill":
        program_config = MLP._get_prefill_pc(seq_len=seq_len, is_w2=False, **cfg["linear_pc_gen"])

    return MLP._fwd_ff1_3(x, cfg["w1"], cfg["w3"], program_config=program_config)


def _run_ds_ff1_3_test(
    mesh_device: ttnn.MeshDevice,
    run_config: dict,
    tt_input: ttnn.Tensor,
    ref_outputs: tuple[torch.Tensor, torch.Tensor],
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
    use_real_weights: bool,
):
    # Log run configuration for superset
    log_run_mode(mode, trace_mode, program_cache_enabled, seq_len, use_real_weights=use_real_weights)

    ref_w1_out, ref_w3_out = ref_outputs
    tt_w1_out, tt_w3_out = ds_ff1_3_ttnn(tt_input, run_config, mode, seq_len)

    tt_w1_out_torch = ttnn.to_torch(
        tt_w1_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_w3_out_torch = ttnn.to_torch(
        tt_w3_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    logger.info("Comparing w1_out (gate projection):")
    w1_pcc, w1_max_abs_error = compare_with_reference(
        tt_w1_out_torch, ref_w1_out, expected_pcc, expected_atol, expected_rtol, strict_assert=False
    )
    logger.info("Comparing w3_out (up projection):")
    w3_pcc, w3_max_abs_error = compare_with_reference(
        tt_w3_out_torch, ref_w3_out, expected_pcc, expected_atol, expected_rtol, strict_assert=False
    )

    if os.getenv(DEVICE_PERF_ENV_VAR) is None:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        warmup_iters = get_int_env("DS_FF1_3_PERF_WARMUP_ITERS", PERF_WARMUP_ITERS)
        measure_iters = get_int_env("DS_FF1_3_PERF_MEASURE_ITERS", PERF_MEASURE_ITERS)
        logger.info(
            f"Starting e2e perf measurement: trace_mode={trace_mode}, program_cache={program_cache_enabled}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}"
        )

        perf_profiler.start("run")
        perf_profiler.start(step_name)

        def op_fn(*, persistent_output_buffer=None):
            return ds_ff1_3_ttnn(tt_input, run_config, mode, seq_len)

        perf_us = measure_perf_us(
            mesh_device,
            op_fn,
            warmup_iters,
            measure_iters,
            trace_mode=trace_mode,
            profiler_name="ds_ff1_3_perf",
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
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-w1_pcc", w1_pcc)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-w1_max_abs_error", w1_max_abs_error)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-w3_pcc", w3_pcc)
        benchmark_data.add_measurement(perf_profiler, 0, step_name, f"{step_name}-w3_max_abs_error", w3_max_abs_error)
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
                "op_type": "ff1_3",
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
            return ds_ff1_3_ttnn(tt_input, run_config, mode, seq_len)

        for _ in range(PERF_WARMUP_ITERS):
            outputs = op_fn()
            ttnn.synchronize_device(mesh_device)
            deallocate_outputs(outputs)

        ttnn.synchronize_device(mesh_device)
        if trace_mode:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            traced_outputs = op_fn()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
            deallocate_outputs(traced_outputs)
        else:
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                outputs = op_fn()
                ttnn.synchronize_device(mesh_device)
                deallocate_outputs(outputs)
            signpost("stop")


def _build_ff1_3_weights(hf_config, use_real_weights: bool) -> FusedWeights:
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
    return FusedWeights(w1=state_dict["gate_proj.weight"], w3=state_dict["up_proj.weight"])


def _build_ff1_3_inputs(
    mesh_device: ttnn.MeshDevice,
    hf_config,
    cache_path: str,
    ccl,
    force_recalculate_weight_config: bool,
    use_real_weights: bool,
    mode: str,
    seq_len: int,
):
    weights = _build_ff1_3_weights(hf_config, use_real_weights)

    state_dict = {
        "gate_proj.weight": weights.w1,
        "up_proj.weight": weights.w3,
        "down_proj.weight": torch.randn(hf_config.hidden_size, hf_config.intermediate_size, dtype=torch.bfloat16),
    }

    from models.demos.deepseek_v3.tt.mlp.mlp import MLP

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
    if mode == "decode":
        torch_input = torch.randn(num_layers, seq_len, batch_size, hf_config.hidden_size, dtype=torch.bfloat16)
    else:
        torch_input = torch.randn(num_layers, batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    x = ttnn.experimental.all_gather_async(tt_input, **ccl.populate_all_gather_runtime_args(run_config["all_gather"]))
    # After sharding with dims=(0, -1), each device has num_layers/mesh_rows layers
    # The reshape must use the per-device layer count, not the total
    num_layers_per_device = x.shape[0]

    effective_seq_len = seq_len
    ref_input = torch_input
    if mode == "prefill" and seq_len > run_config["max_rows"]:
        num_chunks = math.ceil(seq_len / run_config["max_rows"])
        x = ttnn.reshape(x, [num_layers_per_device, num_chunks, run_config["max_rows"], -1])
        # Reference uses full num_layers since we'll concat all device outputs
        ref_input = torch_input.reshape(num_layers, num_chunks, run_config["max_rows"], -1)
        effective_seq_len = run_config["max_rows"]

    ref_outputs = ds_ff1_3_reference(ref_input, weights, mode)
    return run_config, x, ref_outputs, batch_size, effective_seq_len


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # PCC ~0.97 is acceptable for bfloat4_b quantized weights
        # batch_size=32 for all modes
        ("decode", 1, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 128, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 1024, 0.97, 0.5, 0.5, 0.0),
        pytest.param(
            "prefill",
            8192,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(
                os.getenv("DEEPSEEK_V3_LONG_SEQ_TESTS") is None,
                reason="Set DEEPSEEK_V3_LONG_SEQ_TESTS=1 to enable long seq tests",
            ),
        ),
        pytest.param(
            "prefill",
            32768,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(
                os.getenv("DEEPSEEK_V3_LONG_SEQ_TESTS") is None,
                reason="Set DEEPSEEK_V3_LONG_SEQ_TESTS=1 to enable long seq tests",
            ),
        ),
        pytest.param(
            "prefill",
            131072,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(
                os.getenv("DEEPSEEK_V3_LONG_SEQ_TESTS") is None,
                reason="Set DEEPSEEK_V3_LONG_SEQ_TESTS=1 to enable long seq tests",
            ),
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
def test_ds_ff1_3(
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
    is_ci_env,
):
    # CI skip logic: keep only decode/1/trace and prefill/128/eager in CI with program_cache and real_weights
    if is_ci_env:
        ci_keep = (mode == "decode" and seq_len == 1 and trace_mode and program_cache_enabled and use_real_weights) or (
            mode == "prefill" and seq_len == 128 and not trace_mode and program_cache_enabled and use_real_weights
        )
        if not ci_keep:
            pytest.skip("CI test only runs decode/1/trace and prefill/128/eager with program_cache and real_weights")

    if mode == "decode":
        assert seq_len == 1, "Decode only supports seq_len=1"
    else:
        assert mode == "prefill", "Unsupported mode"

    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    run_config, tt_input, ref_outputs, batch_size, effective_seq_len = _build_ff1_3_inputs(
        mesh_device,
        hf_config,
        cache_path,
        ccl,
        force_recalculate_weight_config,
        use_real_weights,
        mode,
        seq_len,
    )
    _run_ds_ff1_3_test(
        mesh_device,
        run_config,
        tt_input,
        ref_outputs,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        mode,
        effective_seq_len,
        batch_size,
        f"ds_ff1_3_{mode}_seq{seq_len}",
        use_real_weights,
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 128, 0.97, 0.5, 0.5, 0.0),
        ("prefill", 1024, 0.97, 0.5, 0.5, 0.0),
        pytest.param(
            "prefill",
            8192,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(
                os.getenv("DEEPSEEK_V3_LONG_SEQ_TESTS") is None,
                reason="Set DEEPSEEK_V3_LONG_SEQ_TESTS=1 to enable long seq tests",
            ),
        ),
        pytest.param(
            "prefill",
            32768,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(
                os.getenv("DEEPSEEK_V3_LONG_SEQ_TESTS") is None,
                reason="Set DEEPSEEK_V3_LONG_SEQ_TESTS=1 to enable long seq tests",
            ),
        ),
        pytest.param(
            "prefill",
            131072,
            0.97,
            0.5,
            0.5,
            0.0,
            marks=pytest.mark.skipif(
                os.getenv("DEEPSEEK_V3_LONG_SEQ_TESTS") is None,
                reason="Set DEEPSEEK_V3_LONG_SEQ_TESTS=1 to enable long seq tests",
            ),
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
def test_ds_ff1_3_single_device(
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
    skip_single_device_sharded("ds_ff1_3")


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
def test_ds_ff1_3_device_perf(mode, seq_len):
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

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_ff1_3_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len}"
    command = f'pytest {test_path}::test_ds_ff1_3 -k "{expr}"'

    profiler.start("run")
    profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = collect_device_perf(
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
        pytest.param("prefill", 1024, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 8192, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 32768, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
        pytest.param("prefill", 131072, marks=pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")),
    ],
)
def test_ds_ff1_3_single_device_device_perf(mode, seq_len):
    skip_single_device_sharded("ds_ff1_3")


if __name__ == "__main__":
    pytest.main([__file__])
