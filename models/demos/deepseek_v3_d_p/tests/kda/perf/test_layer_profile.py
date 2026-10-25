# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-program device timeline of the Kimi-K3 KDA layer on Galaxy SP8xTP4 (a measurement tool, not a gate).

Captures one layer call in a trace and replays it several times under the real-time profiler. For each replay and
chip it records every program's start and end, and writes the JSON to KDA_PROFILE_OUT (default
generated/kda_profile.json): the per-op times, the gaps between programs and the skew between chips.

KDA_PROFILE_TOKENS sets the chunk length (default 5120; multiples of 256 from 1024), KDA_PROFILE_REPLAYS the replays
and KDA_PROFILE_CALLS the calls each replay runs back to back. KDA_PROFILE_DELAY=point:delay_us:chips (see
delay_injection.DelayConfig) profiles the layer with that delay injected, its spin program recorded as one more program.
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
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _L1_SMALL_SIZE,
    _capture_layer_trace,
    _release_layer_trace,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_WORKER_L1_SIZE
from tests.ttnn.profiling.realtime_profiler_utils import require_realtime_profiler

_TOKENS = int(os.environ.get("KDA_PROFILE_TOKENS", "5120"))
_REPLAYS = int(os.environ.get("KDA_PROFILE_REPLAYS", "5"))
# Calls per replay, run back to back without a host sync between them, so the chips reach their steady-state skew.
_CALLS = int(os.environ.get("KDA_PROFILE_CALLS", "4"))
_OUT = Path(os.environ.get("KDA_PROFILE_OUT", "generated/kda_profile.json"))
_DELAY = os.environ.get("KDA_PROFILE_DELAY")
_SP = 8
_SETTLE_SECONDS = 0.5
_TIMEOUT_SECONDS = 20.0


def _replay_records(mesh_device: ttnn.MeshDevice, trace_id: int) -> tuple[float, list[dict]]:
    """Host wall (ms) per call of _CALLS back-to-back replays and every program record they produced, with raw
    timestamps (each chip's records in start order hold the calls one after another)."""
    records: list[dict] = []
    dropped = [0]

    def collect(batch):
        dropped[0] += int(batch.dropped)
        for record in batch.records:
            frequency = float(record.frequency)
            start, end = int(record.start_timestamp), int(record.end_timestamp)
            if not record.runtime_id or frequency <= 0 or end <= start:
                continue
            records.append(
                {
                    "runtime_id": int(record.runtime_id),
                    "chip": int(record.chip_id),
                    "start_ns": start / frequency,
                    "end_ns": end / frequency,
                    "kernels": sorted({Path(str(s)).stem for s in record.kernel_sources}),
                }
            )

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    try:
        begin = time.perf_counter()
        for _ in range(_CALLS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        wall_ms = (time.perf_counter() - begin) * 1e3 / _CALLS
        deadline = time.monotonic() + _TIMEOUT_SECONDS
        last, changed = -1, time.monotonic()
        while time.monotonic() < deadline:
            if len(records) != last:
                last, changed = len(records), time.monotonic()
            elif records and time.monotonic() - changed >= _SETTLE_SECONDS:
                break
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    assert not dropped[0], f"the real-time profiler dropped {dropped[0]} records"
    return wall_ms, records


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
def test_kimi_k3_layer_profile(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Write the per-program timeline of a traced SP8xTP4 layer call at KDA_PROFILE_TOKENS tokens."""
    require_realtime_profiler("the KDA layer profile")
    local_rows = _TOKENS // _SP
    # A full chunk whose start falls mid-device, as the perf gate runs it.
    actual_start = 4 * local_rows + (local_rows // 2) // 32 * 32
    case = make_synthetic_kimi_k3_test_case(sequence=_TOKENS)
    layer, hidden_tt = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    with DelayConfig.parse(_DELAY).inject(layer) if _DELAY else contextlib.nullcontext():
        trace = _capture_layer_trace(mesh_device, layer, hidden_tt, actual_start, actual_start + _TOKENS)
        replays = []
        try:
            _replay_records(mesh_device, trace.trace_id)  # first-replay jitter
            for _ in range(_REPLAYS):
                wall_ms, records = _replay_records(mesh_device, trace.trace_id)
                replays.append({"wall_ms": wall_ms, "records": records})
        finally:
            _release_layer_trace(mesh_device, trace)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "tokens": _TOKENS,
                "local_rows": local_rows,
                "calls_per_replay": _CALLS,
                "actual_start": actual_start,
                "delay": _DELAY,
                # Chip ids in row-major mesh order: index // 4 is the SP rank, index % 4 the TP rank.
                "mesh_device_ids": [int(i) for i in mesh_device.get_device_ids()],
                "replays": replays,
            }
        )
    )
    walls = [r["wall_ms"] for r in replays]
    print(f"KDA_PROFILE tokens={_TOKENS} median_wall_ms={statistics.median(walls):.4f} -> {_OUT}")
