# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sustained traced wall of the Galaxy KDA layer next to each chip's clock and power (a measurement tool, not a gate).

Back-to-back replays push every chip past its power limit, and the firmware then lowers its AI clock. How far depends
on the energy one call takes, so the sustained wall measures energy per call as much as time. This replays the traced
layer for KDA_POWER_SECONDS in blocks of KDA_POWER_BLOCK calls while a thread reads each chip's AI clock, power and
temperature from the kernel driver's sysfs and hwmon files. It reports the steady-state ms per call (second half of
the run) with the mean and minimum AI clock and the mean power over the same window, plus the wall of single calls
after an idle pause. Results go to stdout as a KDA_POWER= line and to KDA_POWER_OUT as JSON.
KDA_POWER_DELAY=point:delay_us:chips[:kind] (see delay_injection.DelayConfig) measures the layer with that delay or
work injected.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from pathlib import Path

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.delay_injection import DelayConfig
from models.demos.deepseek_v3_d_p.tests.kda.perf.telemetry import Telemetry
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _GALAXY_ACTUAL_START,
    _L1_SMALL_SIZE,
    _capture_layer_trace,
    _release_layer_trace,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_WORKER_L1_SIZE
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

_SECONDS = float(os.environ.get("KDA_POWER_SECONDS", "4"))
_BLOCK = int(os.environ.get("KDA_POWER_BLOCK", "10"))
_SINGLES = 10
_IDLE_S = 0.5
_OUT = Path(os.environ.get("KDA_POWER_OUT", "generated/kda_power.json"))
_DELAY = os.environ.get("KDA_POWER_DELAY")


def sustained_run(
    mesh_device: ttnn.MeshDevice, trace_id: int, telemetry: Telemetry, seconds: float, block: int
) -> dict:
    """Idle for _IDLE_S, then replay the trace back to back in blocks of ``block`` calls for ``seconds``. The steady
    state is the run's second half: its median ms per call and the telemetry over it; the idle telemetry is the pause's.
    """
    time.sleep(_IDLE_S)
    blocks = []
    run_start = time.perf_counter()
    while time.perf_counter() - run_start < seconds:
        start = time.perf_counter()
        for _ in range(block):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        end = time.perf_counter()
        blocks.append((start, end, (end - start) * 1e3 / block))
    half = run_start + seconds / 2
    steady = [ms for start, _, ms in blocks if start >= half]
    return {
        "block_calls": block,
        "seconds": seconds,
        "first_blocks_ms_per_call": [ms for _, _, ms in blocks[:3]],
        "steady_ms_per_call": statistics.median(steady),
        "steady_blocks": len(steady),
        "steady_telemetry": telemetry.window(half, blocks[-1][1]),
        "idle_telemetry": telemetry.window(run_start - _IDLE_S + 0.05, run_start),
        "ms_per_call_by_block": [round(ms, 4) for _, _, ms in blocks],
    }


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE, worker_l1_size=KDA_WORKER_L1_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
    ],
    indirect=True,
)
def test_layer_power(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Report the sustained wall per call with the clock and power it ran at."""
    telemetry = Telemetry()
    if not telemetry.available:
        pytest.skip("no Blackhole sysfs telemetry on this host")
    case = make_synthetic_kimi_k3_test_case(sequence=PREFILL_CHUNK_TOKENS)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    actual_start, actual_end = _GALAXY_ACTUAL_START, _GALAXY_ACTUAL_START + PREFILL_CHUNK_TOKENS
    # The injected program's tensors must outlive the trace, so the trace lives inside the injection context.
    with DelayConfig.parse(_DELAY).inject(layer) if _DELAY else contextlib.nullcontext():
        trace = _capture_layer_trace(mesh_device, layer, hidden, actual_start, actual_end)
        telemetry.start()
        try:
            singles_ms = []
            for _ in range(_SINGLES):
                time.sleep(_IDLE_S)
                start = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace.trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                singles_ms.append((time.perf_counter() - start) * 1e3)
            result = {"delay": _DELAY, "single_call_ms": statistics.median(singles_ms)}
            result.update(sustained_run(mesh_device, trace.trace_id, telemetry, _SECONDS, _BLOCK))
        finally:
            telemetry.stop()
            _release_layer_trace(mesh_device, trace)

    print("KDA_POWER=" + json.dumps({key: value for key, value in result.items() if key != "ms_per_call_by_block"}))
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(result))
