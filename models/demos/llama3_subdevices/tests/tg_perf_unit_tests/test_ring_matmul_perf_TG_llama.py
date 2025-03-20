# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

THRESHOLD = 1.0


@pytest.mark.parametrize(
    "mm_type, perf_target_us",
    [
        ("qkv", 7.9),
        ("do", 7.0),
        ("ff13", 8.9),
        ("ff2", 14.8),
        ("lm_head", 380),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ring_mm_tg_llama_perf(
    mm_type,
    perf_target_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"ring_matmul_{mm_type}"

    subdir = "llama_tg_perf"
    command = f"pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul_1d_gather_in0.py::test_matmul_1d_ring_llama_perf -k {mm_type}"
    cols = ["DEVICE KERNEL"]
    op_name = "Matmul"

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(command, subdir, cols, op_name, has_signposts=True)
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_min_us = results[cols[0]]["MIN"] / 1000
    measured_max_us = results[cols[0]]["MAX"] / 1000
    measured_avg_us = results[cols[0]]["AVG"] / 1000
    measured_std_us = results[cols[0]]["STD"] / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"ring_matmul-{mm_type}-min-us", measured_min_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"ring_matmul-{mm_type}-max-us", measured_max_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"ring_matmul-{mm_type}-avg-us", measured_avg_us)
    benchmark_data.add_measurement(profiler, 0, step_name, f"ring_matmul-{mm_type}-std-us", measured_std_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"ring_matmul",
        ml_model_name="llama70b-tg-unit",
    )

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
