# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "warmup_iters, arch_type, perf_target_avg_max_us, perf_target_avg_min_us",
    [
        (5, "T3K", 2500, 2200),
    ],
    ids=["T3K_dit_perf"],
)
def test_all_gather_minimal_perf_t3000(warmup_iters, arch_type, perf_target_avg_max_us, perf_target_avg_min_us):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = "all_gather_minimal_T3K_perf"

    subdir = "ag_perf"
    file = "pytest tests/nightly/t3000/ccl/test_minimal_all_gather_async.py"

    # Target a single minimal all_gather case on T3K with specific ids
    command = (
        file + '::test_all_gather_async -k "dit_shape and perf and barrier_without_persistent_buffers and fabric_ring"'
    )

    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherAsync"

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters)
    profiler.end(step_name)
    profiler.end("run")

    measured_min = results[cols[0]]["MIN"]
    measured_max = results[cols[0]]["MAX"]
    measured_avg = results[cols[0]]["AVG"]
    measured_std = results[cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000
    measured_stddev_us = measured_std / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us with stddev {measured_stddev_us:.3f} us for {op_name}")

    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="all_gather_minimal",
        ml_model_name="ag",
    )

    assert (
        measured_avg_us <= perf_target_avg_max_us
    ), f"Performance target not met due a regression (measured vs expected max): {measured_avg_us} us > {perf_target_avg_max_us} us"
    assert (
        measured_avg_us >= perf_target_avg_min_us
    ), f"Performance target not met due a performance improvement (measured vs expected min): {measured_avg_us} us < {perf_target_avg_min_us} us"
