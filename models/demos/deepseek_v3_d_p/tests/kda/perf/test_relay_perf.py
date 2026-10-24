# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device time of the SP relay alone on Galaxy SP8xTP4 (a measurement tool, not a gate).

Traces RELAY_PERF_CALLS calls of the relay (the model's layout: the right-hand worker columns, no output gate), each
followed by the final carry's SP-line barrier as the model runs it, and replays the trace under the real-time profiler.
Prints, per chronological SP position, the relay program's duration (median over calls, then over the TP chips). The
first position's relay waits for the eighth hop, so its duration is the whole ring's latency; the slope across
positions is the per-hop latency.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import AffineTransform, _RelayFinalCarry, _sp_links, pack_transform
from models.demos.deepseek_v3_d_p.tt.kda.relay_affine_prefix import RelayAffinePrefix
from tests.ttnn.profiling.realtime_profiler_utils import require_realtime_profiler
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

_ROWS, _HEADS, _KEY = 640, 24, 128
_CALLS = int(os.environ.get("RELAY_PERF_CALLS", "8"))
_REPLAYS = int(os.environ.get("RELAY_PERF_REPLAYS", "5"))
_START = int(os.environ.get("RELAY_PERF_START", "0"))
_SETTLE_SECONDS = 0.5
_TIMEOUT_SECONDS = 20.0


def _records(mesh_device, trace_id):
    records, dropped = [], [0]

    def collect(batch):
        dropped[0] += int(batch.dropped)
        for record in batch.records:
            frequency = float(record.frequency)
            start, end = int(record.start_timestamp), int(record.end_timestamp)
            if not record.runtime_id or frequency <= 0 or end <= start:
                continue
            kernels = " ".join(str(s) for s in record.kernel_sources)
            records.append((int(record.chip_id), start / frequency, end / frequency, "relay_affine_prefix" in kernels))

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    try:
        begin = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        wall_us = (time.perf_counter() - begin) * 1e6 / _CALLS
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
    return wall_us, records


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        )
    ],
    indirect=True,
)
def test_relay_perf(mesh_device):
    require_realtime_profiler("the relay perf")
    sp_size, tp_size = tuple(mesh_device.shape)
    generator = torch.Generator().manual_seed(0)

    def per_rank(host):
        return ttnn.from_torch(
            host.reshape((-1,) + host.shape[2:]),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(sp_size, tp_size)),
        )

    a = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
    b = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
    packed = pack_transform(AffineTransform(per_rank(a), per_rank(b)))
    initial = ttnn.from_torch(
        0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    actual_start = make_actual_start(mesh_device, _START)
    grid = mesh_device.compute_with_storage_grid_size()
    relay_columns = [ttnn.CoreCoord(x, y) for x in range(grid.x - 1, grid.x - 4, -1) for y in range(grid.y)]
    links = _sp_links(mesh_device, 0)
    relay = RelayAffinePrefix(
        mesh_device,
        sequence_parallel_axis=0,
        batch_heads=_HEADS,
        key_dim=_KEY,
        value_dim=_KEY,
        num_links=links,
        local_rows=_ROWS,
        cores=relay_columns,
    )
    final_carry = _RelayFinalCarry.shared(mesh_device, 0, links)

    def call():
        entry, final, _ = relay(packed, initial, actual_start)
        carry, _ = final_carry(rank_final, final, actual_start, None, _ROWS)
        return entry, final, carry

    entry, final, _ = relay(packed, initial, actual_start)
    rank_final = ttnn.clone(final)
    call()  # compile
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(_CALLS):
        call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ids = [int(i) for i in mesh_device.get_device_ids()]
    sp_of = {chip: index // tp_size for index, chip in enumerate(ids)}
    first_rank = (_START // _ROWS) % sp_size
    durations = defaultdict(list)  # chip -> relay durations (us), the first call of each replay dropped
    walls = []
    _records(mesh_device, trace_id)
    for _ in range(_REPLAYS):
        wall_us, records = _records(mesh_device, trace_id)
        walls.append(wall_us)
        by_chip = defaultdict(list)
        for chip, start, end, is_relay in records:
            if is_relay:
                by_chip[chip].append((start, end))
        for chip, spans in by_chip.items():
            spans.sort()
            durations[chip] += [(e - s) / 1e3 for s, e in spans[1:]]
    ttnn.release_trace(mesh_device, trace_id)

    per_position = {}
    for chip, values in durations.items():
        position = (sp_of[chip] - first_rank) % sp_size
        per_position.setdefault(position, []).append(statistics.median(values))
    line = " ".join(f"p{p}={statistics.median(v):.1f}" for p, v in sorted(per_position.items()))
    print(f"RELAY_PERF links={links} wall_per_call_us={statistics.median(walls):.1f} relay_us_by_position: {line}")
