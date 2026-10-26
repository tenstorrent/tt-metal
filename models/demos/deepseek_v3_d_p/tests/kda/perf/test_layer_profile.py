# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-program device timeline of the Kimi-K3 KDA layer on Galaxy SP8xTP4 (a measurement tool, not a gate).

Captures one layer call in a trace and replays it several times under the real-time profiler. For each replay and
chip it records every program's start and end, and writes the JSON to KDA_PROFILE_OUT (default
generated/kda_profile.json): the per-op times, the gaps between programs and the skew between chips.

KDA_PROFILE_TOKENS sets the chunk length (default 5120; multiples of 256 from 1024), KDA_PROFILE_REPLAYS the replays
and KDA_PROFILE_CALLS the calls each replay runs back to back. KDA_PROFILE_START moves actual_start (default: mid SP
rank 4). KDA_PROFILE_WARMUP_S replays the trace back to back for that many seconds before each profiled replay, so the
chips run at their sustained, power-limited clocks: profiler durations count cycles, so a chip's cycles per call then
shows its clock relative to the others'.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.profiler_utils import collect_program_records
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _L1_SMALL_SIZE,
    _capture_layer_trace,
    _release_layer_trace,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from models.demos.deepseek_v3_d_p.tt.kda.config import kda_worker_l1_size
from tests.ttnn.profiling.realtime_profiler_utils import require_realtime_profiler

_TOKENS = int(os.environ.get("KDA_PROFILE_TOKENS", "5120"))
_REPLAYS = int(os.environ.get("KDA_PROFILE_REPLAYS", "5"))
# Calls per replay, run back to back without a host sync between them, so the chips reach their steady-state skew.
_CALLS = int(os.environ.get("KDA_PROFILE_CALLS", "4"))
_OUT = Path(os.environ.get("KDA_PROFILE_OUT", "generated/kda_profile.json"))
# Seconds of back-to-back replays before profiling, so the chips reach their sustained (power-limited) clocks.
_WARMUP_SECONDS = float(os.environ.get("KDA_PROFILE_WARMUP_S", "0"))
_SP = 8
_SETTLE_SECONDS = 0.5
_TIMEOUT_SECONDS = 20.0


def _replay_records(mesh_device: ttnn.MeshDevice, trace_id: int) -> tuple[float, list[dict]]:
    """Host wall (ms) per call of _CALLS back-to-back replays and every program record they produced, with raw
    timestamps (each chip's records in start order hold the calls one after another)."""

    def replays():
        for _ in range(_CALLS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

    _, seconds, records = collect_program_records(mesh_device, replays, _SETTLE_SECONDS, _TIMEOUT_SECONDS)
    return seconds * 1e3 / _CALLS, [
        {
            "runtime_id": record.runtime_id,
            "chip": record.chip,
            "start_ns": record.start_ns,
            "end_ns": record.end_ns,
            "kernels": sorted({Path(source).stem for source in record.kernel_sources}),
        }
        for record in records
    ]


def _warm_up(mesh_device: ttnn.MeshDevice, trace_id: int) -> None:
    """Replay the trace back to back for _WARMUP_SECONDS (no profiler callback registered)."""
    deadline = time.monotonic() + _WARMUP_SECONDS
    while time.monotonic() < deadline:
        for _ in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE, worker_l1_size=kda_worker_l1_size),
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
    # KDA_PROFILE_START overrides it (a 32-aligned absolute position), e.g. to move the split to another SP rank.
    actual_start = int(os.environ.get("KDA_PROFILE_START", actual_start))
    case = make_synthetic_kimi_k3_test_case(sequence=_TOKENS)
    layer, hidden_tt = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    trace = _capture_layer_trace(mesh_device, layer, hidden_tt, actual_start, actual_start + _TOKENS)
    replays = []
    try:
        _replay_records(mesh_device, trace.trace_id)  # first-replay jitter
        for _ in range(_REPLAYS):
            _warm_up(mesh_device, trace.trace_id)
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
                # Chip ids in row-major mesh order: index // 4 is the SP rank, index % 4 the TP rank.
                "mesh_device_ids": [int(i) for i in mesh_device.get_device_ids()],
                "replays": replays,
            }
        )
    )
    walls = [r["wall_ms"] for r in replays]
    print(f"KDA_PROFILE tokens={_TOKENS} median_wall_ms={statistics.median(walls):.4f} -> {_OUT}")
