# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

THRESHOLD = 0.4


@pytest.mark.parametrize(
    "warmup_iters, perf_target_us",
    [
        (10, 8.1),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_ag_4_dev(
    warmup_iters,
    perf_target_us,
):
    if ttnn.get_num_devices() != 4:
        pytest.skip("Test is for p150x4 configuration")
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_gather"

    subdir = "llama_ccl_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py -k test_bh_trace_ag"
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

    assert (
        measured_avg_us < perf_target_us + THRESHOLD
    ), f"Performance target not met: {measured_avg_us} us > {perf_target_us} us"
