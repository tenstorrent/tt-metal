# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sustained traced wall of the Galaxy KDA layer next to each chip's clock and power (a measurement tool, not a gate).

Back-to-back replays push every chip past its power limit, and the firmware then lowers its AI clock. How far depends
on the energy one call takes, so the sustained wall measures energy per call as much as time. This replays the traced
layer for KDA_POWER_SECONDS in blocks of KDA_POWER_BLOCK calls while a thread reads each chip's AI clock, power and
temperature from the kernel driver's sysfs and hwmon files. It reports the steady-state ms per call (second half of
the run) with the mean and minimum AI clock and the mean power over the same window, plus the wall of single calls
after an idle pause. Results go to stdout as a KDA_POWER= line and to KDA_POWER_OUT as JSON.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import threading
import time
from pathlib import Path

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
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
_PERIOD_S = 0.01
_SINGLES = 10
_IDLE_S = 0.5
_OUT = Path(os.environ.get("KDA_POWER_OUT", "generated/kda_power.json"))


def _read_int(path: str) -> int | None:
    try:
        with open(path) as file:
            return int(file.read())
    except (OSError, ValueError):
        return None


class _Telemetry(threading.Thread):
    """Samples every Blackhole chip's AI clock (MHz), power (W) and temperature (C) until stopped."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.clocks = sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*/tt_aiclk"))
        self.hwmons = [
            path for path in sorted(glob.glob("/sys/class/hwmon/hwmon*")) if _read_text(f"{path}/name") == "blackhole"
        ]
        self.samples: list[tuple[float, list, list, list]] = []
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            self.samples.append(
                (
                    time.perf_counter(),
                    [_read_int(path) for path in self.clocks],
                    [_scaled(_read_int(f"{path}/power1_input"), 1e6) for path in self.hwmons],
                    [_scaled(_read_int(f"{path}/temp1_input"), 1e3) for path in self.hwmons],
                )
            )
            time.sleep(_PERIOD_S)

    def stop(self) -> None:
        self._done.set()
        self.join()

    def window(self, start: float, end: float) -> dict:
        rows = [row for row in self.samples if start <= row[0] <= end]
        if not rows:
            return {}
        clocks = [value for row in rows for value in row[1] if value is not None]
        powers = [sum(value for value in row[2] if value is not None) for row in rows]
        return {
            "samples": len(rows),
            "aiclk_mean_mhz": statistics.fmean(clocks),
            "aiclk_min_mhz": min(clocks),
            "aiclk_per_chip_mean_mhz": [
                statistics.fmean(row[1][chip] for row in rows) for chip in range(len(self.clocks))
            ],
            "power_total_mean_w": statistics.fmean(powers),
            "power_chip_max_w": max(value for row in rows for value in row[2] if value is not None),
            "temp_max_c": max(value for row in rows for value in row[3] if value is not None),
        }


def _read_text(path: str) -> str | None:
    try:
        with open(path) as file:
            return file.read().strip()
    except OSError:
        return None


def _scaled(value: int | None, unit: float) -> float | None:
    return None if value is None else value / unit


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
    telemetry = _Telemetry()
    if not telemetry.clocks or not telemetry.hwmons:
        pytest.skip("no Blackhole sysfs telemetry on this host")
    case = make_synthetic_kimi_k3_test_case(sequence=PREFILL_CHUNK_TOKENS)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    actual_start, actual_end = _GALAXY_ACTUAL_START, _GALAXY_ACTUAL_START + PREFILL_CHUNK_TOKENS
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

        time.sleep(_IDLE_S)
        blocks = []
        run_start = time.perf_counter()
        while time.perf_counter() - run_start < _SECONDS:
            start = time.perf_counter()
            for _ in range(_BLOCK):
                ttnn.execute_trace(mesh_device, trace.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            end = time.perf_counter()
            blocks.append((start, end, (end - start) * 1e3 / _BLOCK))
    finally:
        telemetry.stop()
        _release_layer_trace(mesh_device, trace)

    half = run_start + _SECONDS / 2
    steady = [ms for start, _, ms in blocks if start >= half]
    first = [ms for _, _, ms in blocks[:3]]
    result = {
        "block_calls": _BLOCK,
        "seconds": _SECONDS,
        "single_call_ms": statistics.median(singles_ms),
        "first_blocks_ms_per_call": first,
        "steady_ms_per_call": statistics.median(steady),
        "steady_blocks": len(steady),
        "steady_telemetry": telemetry.window(half, blocks[-1][1]),
        "idle_telemetry": telemetry.window(run_start - _IDLE_S + 0.05, run_start),
        "ms_per_call_by_block": [round(ms, 4) for _, _, ms in blocks],
    }
    print("KDA_POWER=" + json.dumps({key: value for key, value in result.items() if key != "ms_per_call_by_block"}))
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(result))
