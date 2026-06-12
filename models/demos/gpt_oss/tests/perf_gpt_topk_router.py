# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-perf harness for the fused gpt-oss ``ttnn.experimental.topk_router_gpt`` op.

Spawns ``test_topk_router_perf.py`` (which captures the fused router into a trace and times execute_trace in
the signpost window), reads the per-op device-kernel durations, and logs the per-iter latency. Measurement
only (no perf-target gating). Same measurement recipe as the TTMoEGate perf harness (``perf_tt_moe_gate.py``)
so the gpt-oss fused-router number is directly comparable to TTMoEGate's two-op (linear +
generalized_moe_gate) path on the gpt_oss config.

Run:  pytest models/demos/gpt_oss/tests/perf_gpt_topk_router.py
"""

import pandas as pd
import pytest
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device init in this parent process; the child pytest owns the device."""


def _per_iter_device_kernel_us(subdir: str, num_iters: int) -> tuple[float, float, float]:
    """Per-iter device-kernel latency of the fused router, from the just-generated ops CSV.

    Take ONE chip's rows in the signpost window (single-chip op; this also stays correct if it ever runs on a
    mesh). execute_trace ran num_iters calls back-to-back; split that chip's rows into num_iters equal
    contiguous segments (one call each) and sum each → per-iter totals. Summing whole iterations naturally
    counts every device op the fused kernel emits per call, with no per-op weighting. Returns (avg,min,max) µs."""
    df = pd.read_csv(get_latest_ops_log_filename(subdir))
    markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
    start = markers[markers == "start"].index[0]
    stop = markers[markers == "stop"].index[0]
    window = df.iloc[start + 1 : stop]  # exactly num_iters calls (the child warms up OUTSIDE the window)
    window = window[window["DEVICE ID"] == window["DEVICE ID"].iloc[0]]  # one chip
    dur = pd.to_numeric(window["DEVICE KERNEL DURATION [ns]"], errors="coerce").fillna(0).to_numpy()
    assert (
        len(dur) % num_iters == 0
    ), f"one chip's window has {len(dur)} ops, not divisible by num_iters={num_iters} (calls not uniform?)"
    per_iter_us = dur.reshape(num_iters, -1).sum(axis=1) / 1000  # sum each call's ops → µs per call
    return float(per_iter_us.mean()), float(per_iter_us.min()), float(per_iter_us.max())


@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
@pytest.mark.models_device_performance_bare_metal
def test_topk_router_gpt_perf(warmup_iters, num_iters):
    """Device-perf for the fused gpt-oss router (matmul+bias+topk+softmax, 128 experts, top-4)."""
    subdir = "gpt_oss_router_perf"
    step_name = "topk_router_gpt_128_top4"
    command = "pytest models/demos/gpt_oss/tests/test_topk_router_perf.py::test_topk_router_gpt_perf"
    cols = ["DEVICE KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    profiler.start("run")
    profiler.start(step_name)
    # warmup_iters=0: the child warms up OUTSIDE its signpost window (separate warmup trace), so the window
    # is already exactly num_iters clean calls — trimming rows here would drop real measured ops.
    results = run_device_perf_detailed(command, subdir, cols, "", has_signposts=True, warmup_iters=0, per_op=True)
    profiler.end(step_name)
    profiler.end("run")

    # Headline = per-iter device-kernel latency on one chip (sums whole iterations, so every op the fused
    # kernel emits per call is counted automatically). per-op AVGs are logged as a breakdown.
    measured_avg_us, measured_min_us, measured_max_us = _per_iter_device_kernel_us(subdir, num_iters)
    per_op_us = {op: round(results[op][cols[0]]["AVG"] / 1000, 3) for op in results.keys()}
    logger.info(
        f"[gpt-oss router perf] 128 experts, top-4 (fused matmul+bias+topk+softmax): per-iter device-kernel "
        f"avg={measured_avg_us:.3f} us (min={measured_min_us:.3f}, max={measured_max_us:.3f}) | per-op AVG us={per_op_us}"
    )

    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg-us", measured_avg_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="gpt_oss_router_perf",
        ml_model_name="gpt-oss-120b",
    )


if __name__ == "__main__":
    pytest.main([__file__])
