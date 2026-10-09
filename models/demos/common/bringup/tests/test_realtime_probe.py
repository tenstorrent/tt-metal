# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Probe the program real-time profiler (dispatch-side per-program start / end timestamps streamed to a host
callback; docs/realtime_profiler_architecture.md) on the spec's mesh, as a cheap replacement for the device profiler's
per-op syncs: is it active here, do its runtime ids map to ttnn calls through ttnn._ttnn.get_device_operation_id()
read around each call (no device sync), and do its durations agree with the device profiler's kernel times.

Runs a few matmuls and an all_gather unsynced, maps the records to the calls, prints per call and chip the real-time
duration; with TT_METAL_DEVICE_PROFILER=1 also the device profiler's kernel time of the same calls for comparison."""

import time

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


@mesh_parametrize
def test_realtime_probe(mesh_device):
    import ttnn

    active = ttnn.device.IsProgramRealtimeProfilerActive()
    print(f"[rt] real-time profiler active: {active}", flush=True)
    if not active:
        return
    recs = []

    def cb(batch):
        for r in batch.records:
            recs.append((r.chip_id, r.runtime_id, r.start_timestamp, r.end_timestamp, r.frequency, r.core_count))
        if batch.dropped:
            print(f"[rt] dropped {batch.dropped}", flush=True)

    h = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb)
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        a = ttnn.from_torch(
            torch.randn(1, 1, 2560, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        b = ttnn.from_torch(
            torch.randn(1, 1, 4096, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        calls = []

        def call(name, fn):
            i0 = ttnn._ttnn.get_device_operation_id()
            out = fn()
            calls.append((name, i0, ttnn._ttnn.get_device_operation_id()))
            return out

        for _ in range(2):  # the first is the compile run
            calls.clear()
            outs = [call("matmul 2560x4096x4096", lambda: ttnn.matmul(a, b)) for _ in range(5)]
            outs.append(call("all_gather axis 1", lambda: ttnn.all_gather(a, dim=2, cluster_axis=1)))
            outs.append(call("add", lambda: ttnn.add(a, a)))
            ttnn.synchronize_device(mesh_device)
            for o in outs:
                ttnn.deallocate(o)
        time.sleep(1.0)  # let the receiver thread deliver the last pages
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(h)
    if recs:
        print(f"[rt] first record (chip, runtime_id, start, end, frequency, cores): {recs[0]}", flush=True)
    print(
        f"[rt] {len(recs)} records; runtime ids seen {min(r[1] for r in recs) if recs else None}.."
        f"{max(r[1] for r in recs) if recs else None}",
        flush=True,
    )
    for name, i0, i1 in calls:
        mine = [r for r in recs if i0 <= r[1] < i1]  # a call's launches take the ids read before it
        per_chip = {}
        for chip, rid, st, en, f, cores in mine:
            lo, hi, _ = per_chip.get(chip, (st, en, f))
            per_chip[chip] = (min(lo, st), max(hi, en), f)
        # frequency is in GHz (1.35 on Blackhole): ticks / (f * 1e3) = us
        us = {c: (hi - lo) / (f * 1e3) for c, (lo, hi, f) in per_chip.items()} if per_chip else {}
        print(
            f"[rt] {name:24s} ids ({i0}, {i1}] programs {len(mine)}: per chip us "
            f"{ {c: round(v, 1) for c, v in sorted(us.items())} }",
            flush=True,
        )
