# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-perf harness for the common ``TTMoEGate`` across every model gate config (4×8 mesh, trace-based).

For each model YAML in ``configs/`` that ``TTMoEGate`` can build, spawns
``test_tt_moe_gate_perf.py -k <model>`` (one subprocess per model so each signpost window measures a single
model). The child captures the forward into a trace and times ``execute_trace`` in the window; this harness
reads the per-op device-kernel durations and reports the per-iter latency. Measurement only (no perf-target
gating).

The gate replicates per chip (no cross-chip comms), so ONE chip's device-kernel time IS the per-device
latency — the headline number sums one chip's ops over the window and splits by num_iters (which naturally
counts every op's per-iter multiplicity: Slice ×3, ShardedToInterleaved ×2, ...). Per-op AVGs are logged as
a breakdown.

Run:  pytest models/common/tests/modules/moe/perf_tt_moe_gate.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "modules/moe/configs"


def _gateable_config_ids() -> list[str]:
    """Model configs ``TTMoEGate`` can actually build — mirrors its __init__ guards (= the child's skips):
    n_group ∈ {1, 8}; n_group=8 hardwired to 256 experts select-8. Non-gateable configs are dropped here,
    because the child would only skip them → no signpost window for ``run_device_perf_detailed`` to read."""
    ids = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        config = TTMoEGateConfig.from_yaml(path.read_text())
        if config.n_group not in (1, 8):
            continue
        if config.n_group == 8 and not (config.num_routed_experts == 256 and config.select_experts_k == 8):
            continue
        ids.append(path.stem)
    return ids


GATEABLE_CONFIG_IDS = _gateable_config_ids()
assert GATEABLE_CONFIG_IDS, f"no gateable YAML configs found in {CONFIGS_DIR}"


def _per_iter_device_kernel_us(subdir: str, num_iters: int) -> tuple[float, float, float]:
    """Per-iteration device-kernel latency of ``TTMoEGate.forward``, from the just-generated ops CSV.

    The gate replicates per chip, so every chip runs the same per-device forward — take ONE chip's rows in
    the signpost window (avoids counting the same op ``num_devices`` times). ``execute_trace`` ran num_iters
    forwards back-to-back; split that chip's rows into num_iters equal contiguous segments (one forward each)
    and sum each → per-iter totals. Summing whole forwards naturally counts every op's per-iter multiplicity
    (Slice ×3, ShardedToInterleaved ×2, ...; the ttnn fallback emits none of them) with no per-op weighting.
    Returns (avg, min, max) µs across the num_iters forwards."""
    df = pd.read_csv(get_latest_ops_log_filename(subdir))
    markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
    start = markers[markers == "start"].index[0]
    stop = markers[markers == "stop"].index[0]
    window = df.iloc[start + 1 : stop]  # exactly num_iters forwards (the child warms up OUTSIDE the window)
    window = window[window["DEVICE ID"] == window["DEVICE ID"].iloc[0]]  # one chip → per-device latency
    dur = pd.to_numeric(window["DEVICE KERNEL DURATION [ns]"], errors="coerce").fillna(0).to_numpy()
    assert (
        len(dur) % num_iters == 0
    ), f"one chip's window has {len(dur)} ops, not divisible by num_iters={num_iters} (forwards not uniform?)"
    per_iter_us = dur.reshape(num_iters, -1).sum(axis=1) / 1000  # sum each forward's ops → µs per forward
    return float(per_iter_us.mean()), float(per_iter_us.min()), float(per_iter_us.max())


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device init in this parent process; the child pytest owns the device."""


@pytest.mark.parametrize("model", GATEABLE_CONFIG_IDS)
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
@pytest.mark.models_device_performance_bare_metal
def test_tt_moe_gate_perf(model, warmup_iters, num_iters):
    subdir = "tt_moe_gate_perf"
    step_name = f"tt_moe_gate_{model}"
    # one model per child run: `-k <model>` selects that config's parametrization (stems are unambiguous).
    command = f"pytest models/common/tests/modules/moe/test_tt_moe_gate_perf.py::test_tt_moe_gate_perf -k {model}"
    cols = ["DEVICE KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    profiler.start("run")
    profiler.start(step_name)
    # warmup_iters=0: the child warms up OUTSIDE its signpost window (separate warmup trace), so the window
    # is already exactly num_iters clean forwards — trimming rows here would drop real measured ops.
    results = run_device_perf_detailed(command, subdir, cols, "", has_signposts=True, warmup_iters=0, per_op=True)
    profiler.end(step_name)
    profiler.end("run")

    # Headline = per-iter device-kernel latency of TTMoEGate.forward on one chip (router matmul + gate op +
    # reshapes/slices), summing whole forwards (so every op's per-iter multiplicity is counted automatically).
    measured_avg_us, measured_min_us, measured_max_us = _per_iter_device_kernel_us(subdir, num_iters)
    per_op_us = {op: round(results[op][cols[0]]["AVG"] / 1000, 3) for op in results.keys()}
    logger.info(
        f"[tt_moe_gate perf] {model}: per-iter device-kernel avg={measured_avg_us:.3f} us "
        f"(min={measured_min_us:.3f}, max={measured_max_us:.3f}) | per-op AVG us={per_op_us}"
    )

    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg-us", measured_avg_us)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="tt_moe_gate_perf",
        ml_model_name="tt-moe-gate",
    )


if __name__ == "__main__":
    pytest.main([__file__])
