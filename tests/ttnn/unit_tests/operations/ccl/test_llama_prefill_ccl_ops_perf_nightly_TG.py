# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import os
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed


@pytest.mark.parametrize(
    "shape_id, warmup_iters, perf_target_us",
    [
        # TODO: update when all work on ccls is done
        # 128
        ("128_seq and sh-2048w", 15, 48),
        ("128_seq and sh-1280w", 15, 25),
        ("128_seq and sh-3584w", 15, 50),
        # 4k
        ("4k_seq and sh-2048w", 15, 490),
        ("4k_seq and sh-1280w", 15, 280),
        ("4k_seq and sh-3584w", 15, 750),
        # 8k
        ("8k_seq and sh-2048w", 15, 950),
        ("8k_seq and sh-1280w", 15, 540),
        ("8k_seq and sh-3584w", 15, 1485),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_rs_ring(shape_id, warmup_iters, perf_target_us, galaxy_type):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"rs"

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_prefill_ccl_ops_TG.py::test_reduce_scatter_TG -k \\"{shape_id} and yes-trace\\"'
    else:
        pytest.skip("Needs 6U ring")

    cols = ["DEVICE KERNEL"]
    op_name = "ReduceScatterMinimalAsync"

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
        run_type=f"tg_llama_ops",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"


@pytest.mark.parametrize(
    "shape_id, warmup_iters, perf_target_us",
    [
        # TODO: update when all work on ccls is done
        # 128
        ("128_seq and sh-256w", 15, 26),
        ("128_seq and sh-320w", 15, 20),
        ("128_seq and sh-896w", 15, 30),
        ("128_seq and sh-32w", 15, 16),
        # 4k
        ("4k_seq and sh-256w", 15, 490),
        ("4k_seq and sh-320w", 15, 285),
        ("4k_seq and sh-896w", 15, 770),
        ("4k_seq and sh-32w", 15, 35),
        # 8k
        ("8k_seq and sh-256w", 15, 960),
        ("8k_seq and sh-320w", 15, 550),
        ("8k_seq and sh-896w", 15, 1485),
        ("8k_seq and sh-32w", 15, 65),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_ring(shape_id, warmup_iters, perf_target_us, galaxy_type):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ag"

    subdir = "llama_ccl_perf"
    if galaxy_type == "6U":
        command = f'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_prefill_ccl_ops_TG.py::test_all_gather_TG -k \\"{shape_id} and yes-trace\\"'
    else:
        pytest.skip("Needs 6U ring")

    cols = ["DEVICE KERNEL"]
    op_name = "AllGatherAsync"

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
        run_type=f"tg_llama_ops",
        ml_model_name="llama70b-tg",
    )

    assert measured_avg_us < perf_target_us, f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
