# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

THRESHOLD = 0.4


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device initialization since this test spawns child pytest processes."""


# Override galaxy_type fixture to avoid calling ttnn.cluster.get_cluster_type()
# which might acquire device locks
@pytest.fixture(scope="function")
def galaxy_type():
    """Return galaxy type without initializing devices."""
    import os

    # Use environment variable if set, otherwise default to "TG"
    return os.environ.get("GALAXY_TYPE", "TG")


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("mla_linear_wq_kv_a", 10, 12.54),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_tg_llama_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "llama_ccl_perf"
    command = f"pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_linear_deepseek.py"
    cols = ["DEVICE KERNEL"]
    op_name = "MeshDeviceOperationAdapter<ttnn::operations::matmul::MatmulDeviceOperation>"
    warmup_iters = warmup_iters * 32  # 5 iterations per device

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min = results[cols[0]]["MIN"]
    measured_max = results[cols[0]]["MAX"]
    measured_avg = results[cols[0]]["AVG"]
    measured_std = results[cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_llama_ops_6U",
        ml_model_name="llama70b-tg",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
