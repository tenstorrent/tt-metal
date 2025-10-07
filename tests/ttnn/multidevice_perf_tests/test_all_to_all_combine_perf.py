# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import os
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed


@pytest.mark.parametrize(
    "warmup_iters, arch_type, stage, perf_target_max_us, perf_target_min_us",
    [
        (5, "6U", "decode", 36, 27),
        (5, "TG", "decode", 87, 75),
        (5, "T3K", "decode", 105, 92),  # huge variance among machines for T3K
        (1, "6U", "prefill", 1370, 1300),
        (1, "TG", "prefill", 6200, 5800),
        (1, "T3K", "prefill", 10700, 9800),
    ],
    ids=["6U_decode", "TG_decode", "T3K_decode", "6U_prefill", "TG_prefill", "T3K_prefill"],
)
@pytest.mark.models_device_performance_bare_metal
def test_all_to_all_combine_perf(
    warmup_iters,
    arch_type,
    stage,
    perf_target_max_us,
    perf_target_min_us,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"all_to_all_combine_{arch_type}_{stage}_perf"

    subdir = "moe_perf"
    if arch_type == "6U":
        file = f"pytest tests/ttnn/unit_tests/operations/ccl/test_all_to_all_combine_6U.py"
    elif arch_type == "TG":
        file = f"pytest tests/ttnn/unit_tests/operations/ccl/test_all_to_all_combine_TG.py"
    elif arch_type == "T3K":
        file = f"pytest tests/ttnn/unit_tests/operations/ccl/test_all_to_all_combine_t3000.py"
    else:
        raise ValueError(f"Invalid arch_type: {arch_type}")

    # Unlike dispatch which has separate test_decode_perf and test_prefill_perf functions,
    # combine uses a single test_perf function with decode/prefill as pytest ids
    command = file + f"::test_perf -k {stage}"

    cols = ["DEVICE KERNEL"]
    op_name = "AllToAllCombineDeviceOperation"

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
    measured_max_us = measured_max / 1000

    logger.info(f"Measured performance: {measured_max_us:.3f} us vs. target: {perf_target_max_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"all_to_all_combine",
        ml_model_name="moe",
    )

    assert (
        measured_max_us <= perf_target_max_us
    ), f"Performance target not met due a regression (measured vs expected max): {measured_max_us} us > {perf_target_max_us} us"
    assert (
        measured_max_us >= perf_target_min_us
    ), f"Performance target not met due a performance improvement (measured vs expected min): {measured_max_us} us < {perf_target_min_us} us"
