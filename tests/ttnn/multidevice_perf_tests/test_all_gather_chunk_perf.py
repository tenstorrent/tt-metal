# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import os
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import get_max_chunks_per_sync


@pytest.mark.parametrize("arch_type", ["T3K"])
@pytest.mark.models_device_performance_bare_metal
def test_all_gather_chunk_perf(
    arch_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    subdir = "ag_perf"
    if arch_type == "T3K":
        file = f"pytest tests/nightly/t3000/ccl/test_minimal_all_gather_async.py"
    else:
        raise ValueError(f"Invalid arch_type: {arch_type}")

    base_command = file + "::test_all_gather_chunks_per_sync"

    ag_output_shape = [1, 1, 1024, 5120]
    chunks_per_sync_list = ["MAX", 160, 80, 40, 20, 10, 5, 2, 1]
    for chunks_per_sync in chunks_per_sync_list:
        cols = ["DEVICE KERNEL"]
        op_name = "AllGatherAsync"
        step_name = f"all_gather_chunk_perf_{arch_type}_{chunks_per_sync}_perf"

        final_command = base_command + f" -k '{chunks_per_sync}'"
        profiler.start("run")
        profiler.start(step_name)
        results = run_device_perf_detailed(final_command, subdir, cols, op_name, has_signposts=False, warmup_iters=10)
        profiler.end(step_name)
        profiler.end("run")

        # Get the measured performance
        measured_min = results[cols[0]]["MIN"]
        measured_max = results[cols[0]]["MAX"]
        measured_avg = results[cols[0]]["AVG"]
        measured_std = results[cols[0]]["STD"]
        measured_avg = measured_avg / 1000

        logger.info(f"Measured performance for {chunks_per_sync}: {measured_avg:.3f}")

        # Save the measurement
        benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{chunks_per_sync}-min", measured_min)
        benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{chunks_per_sync}-max", measured_max)
        benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{chunks_per_sync}-avg", measured_avg)
        benchmark_data.add_measurement(profiler, 0, step_name, f"{op_name}-{chunks_per_sync}-std", measured_std)

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"all_gather_chunk_perf",
        ml_model_name="ag",
    )
