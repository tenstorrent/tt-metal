# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed
from tests.ttnn.unit_tests.operations.test_prefetcher_TG import (
    LLAMA_INPUT_SHAPES,
    LLAMA_INPUT_DTYPES,
)

THRESHOLD = 2  # 2 GB/s
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
TILE_HW = 1024


def calculate_bytes(shape, dtype):
    """
    Calculate the number of bytes for a given shape and data type.
    """
    dtype_bytes = TILE_BYTES[dtype]
    return shape[0] * shape[1] / TILE_HW * dtype_bytes


@pytest.mark.parametrize(
    "bw_target",
    [
        200.0,
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_dram_prefetcher_perf(
    bw_target,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"dram_prefetcher"

    subdir = "llama_tg_perf"
    command = f"pytest tests/ttnn/unit_tests/operations/test_prefetcher_TG.py::test_run_prefetcher_llama_perf"
    cols = ["DEVICE NCRISC KERNEL"]
    op_name = "DramPrefetcher"

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(
        command, subdir, cols, op_name, has_signposts=True, device_analysis_types=["device_ncrisc_kernel_duration"]
    )
    profiler.end(step_name)
    profiler.end("run")

    # Get the measured performance
    measured_duration_ns = results[cols[0]]["AVG"]

    # Calculate the effective bandwidth
    # Calculate the total bytes transferred
    total_bytes = [calculate_bytes(shape, dtype) for shape, dtype in zip(LLAMA_INPUT_SHAPES, LLAMA_INPUT_DTYPES)]
    total_bytes = sum(total_bytes)
    effective_bw = total_bytes / measured_duration_ns  # GB/s

    logger.info(f"Measured BW (GB/s): {effective_bw:.3f} vs. target: {bw_target}")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"dram-prefetcher-effective-bw", effective_bw)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"dram_prefetcher",
        ml_model_name="llama70b-tg",
    )

    assert (
        effective_bw + THRESHOLD > bw_target
    ), f"Performance target not met: {effective_bw} GB/s > {bw_target} GB/s within {THRESHOLD} GB/s"
