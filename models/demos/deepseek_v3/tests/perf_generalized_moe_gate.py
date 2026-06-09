# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-perf harness for the fused ``generalized_moe_gate`` op (256 experts, sigmoid ON).

Spawns ``test_generalized_moe_gate_perf.py``, reads the per-op device-kernel durations from the
signpost-bracketed main trace, and logs the measured µs. Measurement only (no perf-target gating).

Run:  pytest models/demos/deepseek_v3/tests/perf_generalized_moe_gate.py
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
def test_generalized_moe_gate_perf(warmup_iters, num_iters):
    subdir = "deepseek_gmg_perf"
    step_name = "generalized_moe_gate_256_sigmoid"
    command = "pytest models/demos/deepseek_v3/tests/test_generalized_moe_gate_perf.py::test_generalized_moe_gate_perf"
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

    # Sum all device ops seen in the window (the gate is one GeneralizedMoeGateDeviceOperation per
    # iter; stay tolerant of the op name / any pre/post ops).
    measured_avg = sum(results[op][cols[0]]["AVG"] for op in results.keys())
    measured_min = sum(results[op][cols[0]]["MIN"] for op in results.keys())
    measured_max = sum(results[op][cols[0]]["MAX"] for op in results.keys())
    measured_avg_us = measured_avg / 1000

    logger.info(
        f"[gate perf] 256 experts, sigmoid ON: avg={measured_avg_us:.3f} us "
        f"(min={measured_min / 1000:.3f}, max={measured_max / 1000:.3f}) over ops={list(results.keys())}"
    )

    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg-us", measured_avg_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="deepseek_gmg_perf",
        ml_model_name="deepseek-v3",
    )


@pytest.mark.parametrize("topk", [8, 4])
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
@pytest.mark.models_device_performance_bare_metal
def test_generalized_moe_gate_perf_512(warmup_iters, num_iters, topk):
    """Device-perf for the global top-`topk` over 512 experts (the A2 combine path)."""
    subdir = "deepseek_gmg_perf"
    step_name = f"generalized_moe_gate_512_top{topk}_sigmoid"
    # -k selects the matching inner topk variant (ids "top8"/"top4") so only one trace is measured.
    command = (
        "pytest models/demos/deepseek_v3/tests/test_generalized_moe_gate_perf.py::test_generalized_moe_gate_perf_512"
        f' -k "top{topk}"'
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

    # Sum all device ops seen in the window (one GeneralizedMoeGateDeviceOperation per iter).
    measured_avg = sum(results[op][cols[0]]["AVG"] for op in results.keys())
    measured_min = sum(results[op][cols[0]]["MIN"] for op in results.keys())
    measured_max = sum(results[op][cols[0]]["MAX"] for op in results.keys())
    measured_avg_us = measured_avg / 1000

    logger.info(
        f"[gate perf] 512 experts (global top-{topk}, combine), sigmoid ON: avg={measured_avg_us:.3f} us "
        f"(min={measured_min / 1000:.3f}, max={measured_max / 1000:.3f}) over ops={list(results.keys())}"
    )

    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg-us", measured_avg_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="deepseek_gmg_perf",
        ml_model_name="deepseek-v3",
    )


if __name__ == "__main__":
    pytest.main([__file__])
