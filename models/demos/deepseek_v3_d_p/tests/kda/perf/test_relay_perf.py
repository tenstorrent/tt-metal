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
from collections import defaultdict

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.profiler_utils import collect_program_records
from models.demos.deepseek_v3_d_p.tests.kda.utils import RELAY_ROWS, RelayTransforms, make_relay, relay_worker_columns
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import _RelayFinalCarry
from tests.ttnn.profiling.realtime_profiler_utils import require_realtime_profiler
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

_CALLS = int(os.environ.get("RELAY_PERF_CALLS", "8"))
_REPLAYS = int(os.environ.get("RELAY_PERF_REPLAYS", "5"))
_START = int(os.environ.get("RELAY_PERF_START", "0"))
_SETTLE_SECONDS = 0.5
_TIMEOUT_SECONDS = 20.0


def _records(mesh_device, trace_id):
    """Host wall (us) per call of one trace replay, and (chip, start ns, end ns, is the relay) per program."""
    _, seconds, records = collect_program_records(
        mesh_device,
        lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False),
        _SETTLE_SECONDS,
        _TIMEOUT_SECONDS,
    )
    return seconds * 1e6 / _CALLS, [
        (
            record.chip,
            record.start_ns,
            record.end_ns,
            any("relay_affine_prefix" in source for source in record.kernel_sources),
        )
        for record in records
    ]


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
    transforms = RelayTransforms(mesh_device, seed=0)
    packed, initial = transforms.packed, transforms.initial_tt
    actual_start = make_actual_start(mesh_device, _START)
    relay = make_relay(mesh_device, cores=relay_worker_columns(mesh_device))
    final_carry = _RelayFinalCarry.shared(mesh_device, 0, relay.num_links)

    _, final, _ = relay(packed, initial, actual_start)
    # rank_final stands in for the correction's output: the carry only needs a tensor of the right shape.
    rank_final = ttnn.clone(final)

    def call():
        entry, final, _ = relay(packed, initial, actual_start)
        carry, _ = final_carry(rank_final, final, actual_start, None, RELAY_ROWS)
        return entry, final, carry

    call()  # compile
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(_CALLS):
        call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ids = [int(i) for i in mesh_device.get_device_ids()]
    sp_of = {chip: index // tp_size for index, chip in enumerate(ids)}
    first_rank = (_START // RELAY_ROWS) % sp_size
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
    print(
        f"RELAY_PERF links={relay.num_links} wall_per_call_us={statistics.median(walls):.1f} relay_us_by_position: {line}"
    )
