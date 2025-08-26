# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
from loguru import logger
import os
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import get_max_chunks_per_sync


def total_elems(ag_output_shape):
    return math.prod(ag_output_shape)


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

    num_devices = 8
    output_shapes = [[1, 1, 1024, 5120]]
    chunks_per_sync_list = ["MAX", 160]
    for i, ag_output_shape in enumerate(output_shapes):
        elements = total_elems(ag_output_shape)
        total_bytes = elements * 2
        data_size_bytes_gb = total_bytes / (10**9)
        logger.info(f"Total elements: {elements}, Data size: {data_size_bytes_gb:.3f} GB")

        for chunks_per_sync in chunks_per_sync_list:
            cols = ["DEVICE KERNEL"]
            op_name = "AllGatherAsync"
            step_name = f"all_gather_chunk_perf_{arch_type}_{chunks_per_sync}_perf"

            # Filter by both chunks_per_sync and shape
            shape_str = f"ag_output_shape{i}"
            final_command = base_command + f' -k "{chunks_per_sync}-chunks and {shape_str}"'
            profiler.start("run")
            profiler.start(step_name)
            results = run_device_perf_detailed(
                final_command, subdir, cols, op_name, has_signposts=False, warmup_iters=10
            )
            profiler.end(step_name)
            profiler.end("run")

            # Get the measured performance
            measured_min = results[cols[0]]["MIN"]
            measured_max = results[cols[0]]["MAX"]
            measured_avg = results[cols[0]]["AVG"]
            measured_std = results[cols[0]]["STD"]

            final_chunks_per_sync = (
                get_max_chunks_per_sync(num_devices, ag_output_shape) if chunks_per_sync == "MAX" else chunks_per_sync
            )
            logger.info(
                f"Measured performance for data size {data_size_bytes_gb:.3f} GB, chunks per sync {final_chunks_per_sync}: {measured_avg/1000:.3f} us at {total_bytes/measured_avg:.6f} GB/s"
            )

            # Save the measurement
            benchmark_data.add_measurement(
                profiler, 0, step_name, f"{op_name}-{final_chunks_per_sync}-min", measured_min
            )
            benchmark_data.add_measurement(
                profiler, 0, step_name, f"{op_name}-{final_chunks_per_sync}-max", measured_max
            )
            benchmark_data.add_measurement(
                profiler, 0, step_name, f"{op_name}-{final_chunks_per_sync}-avg", measured_avg
            )
            benchmark_data.add_measurement(
                profiler, 0, step_name, f"{op_name}-{final_chunks_per_sync}-std", measured_std
            )

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"all_gather_chunk_perf",
        ml_model_name="ag",
    )
