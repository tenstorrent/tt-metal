# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-perf harness for the generic ttnn.topk baseline (top-8 over N experts, single device).

Spawns ``test_topk_experts_perf.py`` (one experts-count at a time), reads the per-op device-kernel
durations from the signpost-bracketed main trace, logs µs per size. The number to beat with the
fused gate. Run:  pytest models/demos/deepseek_v3/tests/perf_topk_experts.py
"""

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device init in this parent process; the child pytest owns the device."""


@pytest.mark.parametrize("num_experts", [256, 512])
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
@pytest.mark.models_device_performance_bare_metal
def test_topk_experts_perf(num_experts, warmup_iters, num_iters):
    subdir = "deepseek_topk_experts_perf"
    step_name = f"topk_{num_experts}"
    command = (
        f"pytest models/demos/deepseek_v3/tests/test_topk_experts_perf.py::test_topk_experts_perf -k {num_experts}"
    )
    cols = ["DEVICE KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(
        command, subdir, cols, "", has_signposts=True, warmup_iters=warmup_iters, per_op=True
    )
    profiler.end(step_name)
    profiler.end("run")

    total_avg = 0.0
    for op in results.keys():
        op_avg_us = results[op][cols[0]]["AVG"] / 1000
        total_avg += op_avg_us
        logger.info(f"[topk] experts={num_experts}  op={op}: {op_avg_us:.3f} us")
    logger.info(f"[topk] experts={num_experts}  TOTAL: {total_avg:.3f} us  (ops={list(results.keys())})")

    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-total-us", total_avg)
    benchmark_data.save_partial_run_json(profiler, run_type="deepseek_topk_experts_perf", ml_model_name="deepseek-v3")


if __name__ == "__main__":
    pytest.main([__file__])
