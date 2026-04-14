# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from loguru import logger

from models.demos.deepseek_v3.tests.fused_op_unit_tests.test_utils import collect_device_perf
from models.demos.deepseek_v3.tests.test_mla import run_test_forward_pass_mla2d
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_fabric_config
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

DEVICE_PERF_ENV_VAR = "DS_MLA_FORWARD_DEVICE_PERF"
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 1.0
DEVICE_PERF_TARGETS_US = {
    ("decode", 1): {"kernel": 0.0, "op_to_op": None},
    ("prefill", 128): {"kernel": 0.0, "op_to_op": None},
}


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
    ],
)
@pytest.mark.parametrize("module_path", [pytest.param("model.layers.0.self_attn", id="real_weights")])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
def test_ds_mla_forward(
    device_params,
    mode,
    seq_len,
    module_path,
    program_cache_enabled,
    trace_mode,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    del set_deterministic_env
    del device_params

    if trace_mode and not program_cache_enabled:
        pytest.skip("Trace mode requires program cache enabled (skip trace + no_program_cache).")

    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    perf_mode_enabled = os.getenv(DEVICE_PERF_ENV_VAR) is not None
    if perf_mode_enabled:
        logger.info(f"[{DEVICE_PERF_ENV_VAR}] MLA perf mode enabled")

    batch_size_per_row = USERS_PER_ROW if mode == "decode" else 1
    run_test_forward_pass_mla2d(
        layer_idx=0,
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=batch_size_per_row,
        hf_config_short=hf_config_short,
        cache_path=cache_path,
        mesh_device=mesh_device,
        ccl=ccl,
        model_path=model_path,
        module_path=module_path,
        force_recalculate_weight_config=force_recalculate_weight_config,
        state_dict=state_dict,
        decode_position_ids=None,
        perf_mode=perf_mode_enabled,
        trace_mode=trace_mode,
        num_iters=DEVICE_PERF_ITERS,
    )


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
    ],
)
@pytest.mark.timeout(1800)
def test_ds_mla_forward_device_perf(mode, seq_len):
    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    batch_size = USERS_PER_ROW * mesh_shape[0] if mode == "decode" else 1

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ds_mla_forward_device_perf_{mode}_seq{seq_len}"
    test_path = "models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_mla.py"
    trace_filter = "trace" if mode == "decode" else "eager"
    expr = (
        f"test_ds_mla_forward and {mode} and {seq_len} and real_weights and "
        f"{trace_filter} and program_cache and not no_program_cache"
    )
    command = f'pytest {test_path}::test_ds_mla_forward -k "{expr}"'

    perf_profiler.start("run")
    perf_profiler.start(step_name)
    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    op_stats, total_kernel_ns, total_op_to_op_ns = collect_device_perf(
        command,
        subdir="deepseek_v3_fused_ops_device_perf",
        warmup_iters=0,
        use_signposts=False,
    )
    os.environ.pop(DEVICE_PERF_ENV_VAR, None)
    perf_profiler.end(step_name)
    perf_profiler.end("run")

    assert op_stats, "No device perf stats captured."
    total_kernel_us = total_kernel_ns / 1000.0
    total_op_to_op_us = total_op_to_op_ns / 1000.0
    logger.info(f"Device perf per-op averages (ns): {json.dumps(op_stats, indent=2)}")
    logger.info(f"Device perf totals: kernel={total_kernel_us:.3f} us, op_to_op={total_op_to_op_us:.3f} us")
    print(f"[DS_MLA_PERF] op_to_op_latency_us={total_op_to_op_us:.3f}, kernel_time_us={total_kernel_us:.3f}")
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
