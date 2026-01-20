# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed


@pytest.mark.parametrize(
    "warmup_iters, perf_target_us, threshold",
    [
        (10, 100.0, 0.5),  # Placeholder target
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_all_gather_wan_perf(
    warmup_iters,
    perf_target_us,
    threshold,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = "all_gather_wan"

    subdir = "wan_ccl_perf"
    command = "pytest tests/nightly/tg/ccl/test_minimal_all_gather_async.py::test_all_gather_wan"
    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherDeviceOperation"
    warmup_iters = warmup_iters * 32  # iterations per device

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
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="tg_wan_ops",
        ml_model_name="wan2_2-tg",
    )

    # assert (
    #    measured_avg_us < perf_target_us + threshold
    # ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


if __name__ == "__main__":
    test_all_gather_wan_perf(
        warmup_iters=10,
        perf_target_us=100.0,
        threshold=0.5,
    )
