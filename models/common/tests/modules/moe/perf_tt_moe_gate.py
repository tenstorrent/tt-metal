# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-perf harness for the common ``TTMoEGate`` on the GPT-OSS gate config.

Spawns ``test_tt_moe_gate_perf.py``, reads the per-op device-kernel durations from the signpost-bracketed
window, and logs the measured µs. Measurement only (no perf-target gating). The summed device-kernel time
covers TTMoEGate's whole forward (router ttnn.linear + generalized_moe_gate + slice/view) per iter — i.e.
the cost of producing the GPT-OSS routing via TTMoEGate's two-op path (vs GPT-OSS's fused topk_router_gpt).

Run:  pytest models/common/tests/modules/moe/perf_tt_moe_gate.py
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
def test_tt_moe_gate_perf(warmup_iters, num_iters):
    subdir = "tt_moe_gate_perf"
    step_name = "tt_moe_gate_gpt_oss_top4"
    command = "pytest models/common/tests/modules/moe/test_tt_moe_gate_perf.py::test_tt_moe_gate_perf"
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

    # Sum the per-op device-kernel time over the window → per-iter cost of TTMoEGate.forward (router
    # matmul + gate op + reshapes/slices). Each op's AVG is averaged over its num_iters occurrences.
    measured_avg = sum(results[op][cols[0]]["AVG"] for op in results.keys())
    measured_min = sum(results[op][cols[0]]["MIN"] for op in results.keys())
    measured_max = sum(results[op][cols[0]]["MAX"] for op in results.keys())
    measured_avg_us = measured_avg / 1000

    logger.info(
        f"[tt_moe_gate perf] gpt_oss (128 experts, top-4, softmax + router bias): avg={measured_avg_us:.3f} us "
        f"(min={measured_min / 1000:.3f}, max={measured_max / 1000:.3f}) over ops={list(results.keys())}"
    )

    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg-us", measured_avg_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="tt_moe_gate_perf",
        ml_model_name="tt-moe-gate",
    )


if __name__ == "__main__":
    pytest.main([__file__])
