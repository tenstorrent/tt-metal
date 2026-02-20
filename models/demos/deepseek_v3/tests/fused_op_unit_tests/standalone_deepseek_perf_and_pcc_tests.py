# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

os.environ.setdefault("MESH_DEVICE", "TG")

THRESHOLD = 0.5
THRESHOLD_PERCENTAGE = 0.03


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device initialization since this test spawns child pytest processes."""


# Override galaxy_type fixture to avoid calling ttnn.cluster.get_cluster_type()
# which might acquire device locks
@pytest.fixture(scope="function")
def galaxy_type():
    """Return galaxy type without initializing devices."""

    # Use environment variable if set, otherwise default to "TG"
    return os.environ.get("GALAXY_TYPE", "TG")


@pytest.mark.parametrize(
    "step_name, warmup_iters, perf_target_us",
    [
        ("rms_norm", 10, 16.27),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_rms_norm_perf_and_pcc(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "deepseek_ccl_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py -k test_6u_trace_rms_fuse_deepseek"
    cols = ["DEVICE KERNEL"]
    op_name = "RMSAllGatherDeviceOperation"

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
        run_type=f"tg_deepseek_linear",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"
