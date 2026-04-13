# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from copy import deepcopy

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import collect_device_perf
from models.demos.deepseek_v3.tests.test_moe import generate_reference_io
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_fabric_config, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
    system_name_to_mesh_shape,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

DEVICE_PERF_ENV_VAR = "DS_MOE_FORWARD_DEVICE_PERF"
DEVICE_PERF_WARMUP_ITERS = 3
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 1.0
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 0.0, "op_to_op": None},
    ("prefill", 128): {"kernel": 0.0, "op_to_op": None},
}


@pytest.fixture
def reference_model(hf_config):
    torch.use_deterministic_algorithms(True)
    moe_config = deepcopy(hf_config)
    moe_config.n_shared_experts = None
    return DeepseekV3MoE(moe_config).eval()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": get_fabric_config(), "trace_region_size": 2967552},
    ],
    indirect=True,
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
@pytest.mark.parametrize("weight_type", [pytest.param("real", id="real_weights")])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
def test_ds_moe_forward(
    device_params,
    mode,
    seq_len,
    weight_type,
    program_cache_enabled,
    trace_mode,
    set_deterministic_env,
    reference_model,
    hf_config,
    request,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
):
    del set_deterministic_env
    del weight_type

    # Trace capture replays pre-compiled binaries. Without program cache,
    # trace can trigger forbidden compile/program writes during capture.
    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    perf_mode_enabled = os.getenv(DEVICE_PERF_ENV_VAR) is not None
    if perf_mode_enabled:
        logger.info(
            f"[{DEVICE_PERF_ENV_VAR}] Device perf mode enabled; running initial setup and correctness pass first"
        )

    module_path = "model.layers.3.mlp"
    checkpoint_state_dict = request.getfixturevalue("state_dict")
    num_tokens = USERS_PER_ROW * mesh_device.shape[0] if mode == "decode" else seq_len
    if perf_mode_enabled:
        # Perf runs should not depend on reference IO cache files or host-side PCC checks.
        # Keep real MoE weights (layer-scoped), but use synthetic input for faster bring-up.
        state_dict = {
            name: tensor
            for name, tensor in sub_state_dict(checkpoint_state_dict, module_path + ".").items()
            if not name.startswith("shared_experts.") and not name.endswith(".weight_scale_inv")
        }
        if not state_dict:
            pytest.skip(f"Checkpoint does not contain routed MoE weights under '{module_path}'")
        torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)
        reference_output = None
        logger.info(
            f"[{DEVICE_PERF_ENV_VAR}] Using synthetic input in perf mode (skipping reference IO cache dependency)"
        )
    else:
        state_dict, torch_input, reference_output = generate_reference_io(
            mode=mode,
            num_tokens=num_tokens,
            reference_model=reference_model,
            hf_config=hf_config,
            weight_type="real",
            checkpoint_state_dict=checkpoint_state_dict,
            module_path=module_path,
        )

    weight_config = get_test_weight_config(
        MoE,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=force_recalculate_weight_config,
        test_name="test_ds_moe_forward",
        real_weights=True,
        layer_id=module_path,
    )

    # MoE gate topk fallback performs host IO (to_torch/from_torch), which is illegal during trace capture.
    # Match fused-op trace behavior by forcing pure device topk path when trace_mode is enabled.
    model_config = get_model_config(
        MoE, mode, hf_config, mesh_device, device_params["fabric_config"], topk_fallback=not trace_mode
    )
    model_state = MoE.create_state(hf_config, mesh_device, ccl)
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    def build_tt_input():
        tt_input = ttnn.from_torch(
            torch_input.unsqueeze(1),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        return ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    def run_op():
        tt_input = build_tt_input()
        tt_output = run_module_forward(MoE, mode, tt_input, run_config, handle_tensor_parallel=True)
        return tt_input, tt_output

    tt_input, tt_output = run_op()
    if perf_mode_enabled:
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)
        logger.info(f"Mode: {mode}, Num tokens: {num_tokens}, Weight type: real")
        logger.info(f"[{DEVICE_PERF_ENV_VAR}] Skipping correctness PCC check in perf mode")
    else:
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)
        logger.info(f"Mode: {mode}, Num tokens: {num_tokens}, Weight type: real")
        assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.97)

    if perf_mode_enabled:
        from tracy import signpost

        def eager_op_fn():
            local_tt_input, local_tt_output = run_op()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(local_tt_input)
            ttnn.deallocate(local_tt_output)

        logger.info(f"[{DEVICE_PERF_ENV_VAR}] Correctness pass complete; entering profiled iteration loops")
        logger.info(f"[{DEVICE_PERF_ENV_VAR}] Starting warmup phase with {DEVICE_PERF_WARMUP_ITERS} iteration(s)")
        if trace_mode:
            trace_input = build_tt_input()
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            trace_output = run_module_forward(MoE, mode, trace_input, run_config, handle_tensor_parallel=True)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            for _ in range(DEVICE_PERF_WARMUP_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            logger.info(f"[{DEVICE_PERF_ENV_VAR}] Starting perf measurement with {DEVICE_PERF_ITERS} iteration(s)")
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                ttnn.execute_trace(mesh_device, trace_id, blocking=False)
                ttnn.synchronize_device(mesh_device)
            signpost("stop")
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.deallocate(trace_input)
            ttnn.deallocate(trace_output)
        else:
            for _ in range(DEVICE_PERF_WARMUP_ITERS):
                eager_op_fn()

            ttnn.synchronize_device(mesh_device)
            logger.info(f"[{DEVICE_PERF_ENV_VAR}] Starting perf measurement with {DEVICE_PERF_ITERS} iteration(s)")
            signpost("start")
            for _ in range(DEVICE_PERF_ITERS):
                eager_op_fn()
            signpost("stop")


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
    ],
)
@pytest.mark.timeout(1800)
def test_ds_moe_forward_device_perf(mode, seq_len):
    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0] if mode == "decode" else 1

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_moe_forward_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = f"program_cache and not no_program_cache and {trace_filter} and {mode} and {seq_len} and real_weights"
    command = f'pytest {test_path}::test_ds_moe_forward -k "{expr}"'

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
        kernel_limit_us = kernel_target_us * (1 + DEVICE_PERF_MARGIN)
        assert (
            total_kernel_us <= kernel_limit_us
        ), f"Kernel perf regression: {total_kernel_us:.3f}us exceeds {kernel_target_us:.3f}us (+{DEVICE_PERF_MARGIN:.0%})"
        op_to_op_target_us = targets.get("op_to_op")
        if op_to_op_target_us is None:
            logger.info("Op-to-op perf target is unset; skipping op-to-op perf assertion.")
        else:
            op_to_op_limit_us = op_to_op_target_us * (1 + DEVICE_PERF_MARGIN)
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
