# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf harness for the fused gpt-oss ``ttnn.experimental.topk_router_gpt`` op.

Spawns ``test_topk_router_gpt_perf.py``, reads the per-op device-kernel durations from the
signpost-bracketed main trace, and logs the measured µs. Measurement only (no perf-target gating).
Mirrors ``perf_generalized_moe_gate.py`` so the gpt-oss router number is directly comparable to the
deepseek generalized_moe_gate number.

Run:  pytest models/demos/gpt_oss/tests/perf_topk_router_gpt.py
"""

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device init in this parent process; the child pytest owns the device."""


@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
@pytest.mark.models_device_performance_bare_metal
def test_topk_router_gpt_perf(warmup_iters, num_iters):
    """Device-perf for the fused gpt-oss router (matmul+bias+topk+softmax, 128 experts, top-4)."""
    subdir = "gpt_oss_router_perf"
    step_name = "topk_router_gpt_128_top4"
    command = "pytest models/demos/gpt_oss/tests/test_topk_router_gpt_perf.py::test_topk_router_gpt_perf"
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

    # Sum all device ops seen in the signpost window (the fused router is one device op per iter; stay
    # tolerant of the op name / any pre/post ops in the window).
    measured_avg = sum(results[op][cols[0]]["AVG"] for op in results.keys())
    measured_min = sum(results[op][cols[0]]["MIN"] for op in results.keys())
    measured_max = sum(results[op][cols[0]]["MAX"] for op in results.keys())
    measured_avg_us = measured_avg / 1000

    logger.info(
        f"[gpt-oss router perf] 128 experts, top-4 (matmul+bias+topk+softmax): avg={measured_avg_us:.3f} us "
        f"(min={measured_min / 1000:.3f}, max={measured_max / 1000:.3f}) over ops={list(results.keys())}"
    )

    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg-us", measured_avg_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="gpt_oss_router_perf",
        ml_model_name="gpt-oss-120b",
    )


if __name__ == "__main__":
    pytest.main([__file__])
