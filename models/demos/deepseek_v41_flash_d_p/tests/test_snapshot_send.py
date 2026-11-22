# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.65 (CHEAP_CCL=on): the window snapshot as one prepared broadcast (tt/snapshot_send.SnapshotSend) against the
legacy path (two ring gathers + slices + concat + update_padded_kv_cache), byte for byte on every chip, and the host
us a call / device us a call of both.

Two window caches (WINDOW_KV_KEEP=snapshot128), each with its own ring, sharing one served slab tensor of LAYERS
(slot, layer) batches (so the program's page offset is exercised: cache A writes batch 1, cache B batch 2; batch 0 is
a guard that must keep its bytes). Both caches take the same random rows through the same ring write path
(``write``: every chip the chunk's rows, blocks held by every chip; ``write_own``: each chip its own query rows, blocks
held by one chip), then A writes its snapshot under CHEAP_CCL=legacy and B under on, B's batch poisoned with random
rows first. Each batch must equal the definition (row p % 128 = position p for p in [cap - 128, cap), zeros for no
position) on every chip. Ends as tests/test_window_snapshot.py (cap < 128, a whole chunk, a mid-chunk end over two owner chips, a follow-up off
the chunk period, owned blocks with an end mid-record, cap 0).

Then host us a call (PROBE_RUNS alternating calls a side, median), back-to-back wall us a call (40 calls then one sync)
and device us a call (real-time profiler: each program's slowest chip, summed), on the whole-chunk owned pattern.

    scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_snapshot_send.py -s
"""

from __future__ import annotations

import os
import statistics
import time

import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

WIDTH = 512
LAYERS = 3
RUNS = int(os.environ.get("PROBE_RUNS", "30"))


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setenv(settings.env_name(k), str(v))
    settings.refresh()


@mesh_parametrize(timeout_s=1800)
def test_snapshot_send(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt.attention import (
        SNAP,
        SP_AXIS,
        TtWindowCache,
        WindowTables,
        _upload,
        snapshot_cap,
    )
    from models.demos.deepseek_v41_flash_d_p.tt.served import CacheSlab, alloc_served

    _set(monkeypatch, WINDOW_KV_KEEP="snapshot128", KV_BLOCK_SIZE=64, CHEAP_CCL="on")
    mesh = mesh_device
    n = mesh.get_num_devices()
    if n < 2:
        import pytest

        pytest.skip("the snapshot send needs at least two chips")
    chunk = int(spec().get("target.chunk"))
    tables = WindowTables(mesh, chunk)
    g = tables.geo
    slab_t = alloc_served(mesh, WIDTH, SNAP, 32, LAYERS, dedup=False)
    caches = {
        side: TtWindowCache(mesh, tables, 4 * chunk, slab=CacheSlab(slab_t, slot=0, row=b, layers=LAYERS))
        for side, b in (("legacy", 1), ("on", 2))
    }
    assert tables.snapshot_send() is not None
    gen = torch.Generator().manual_seed(65)
    full = (torch.randn(4 * chunk, WIDTH, generator=gen) * 2).to(torch.bfloat16)

    def host_batches():
        out = []
        for t in ttnn.get_device_tensors(slab_t):
            out.append(ttnn.to_torch(t).reshape(LAYERS, SNAP, WIDTH))
        return out

    guard = host_batches()

    def nat_rows(start, end):
        nat = torch.zeros(chunk, WIDTH, dtype=torch.bfloat16)
        nat[: end - start] = full[start:end]
        return nat

    def write(how, start, end):
        for c in caches.values():
            if how == "rep":
                t = _upload(mesh, nat_rows(start, end).view(1, 1, chunk, WIDTH), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                c.write(t, start, end)
            else:
                nat = nat_rows(start, end).view(g.sp, g.tp, g.q_rows, WIDTH)
                t = _upload(mesh, nat, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
                c.write_own(t, start, end)
            ttnn.deallocate(t)

    def poison(c):
        rows = _upload(
            mesh,
            (torch.randn(1, 1, SNAP, WIDTH, generator=gen) * 3).to(torch.bfloat16),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        s = c.snap
        ttnn.bringup.update_padded_kv_cache(
            s.tensor,
            rows,
            slot_idx=s.slot,
            layer_idx=s.row,
            num_layers=s.layers,
            kv_actual_global=0,
            cluster_axis=SP_AXIS,
            max_grid=c.ring.grid,
        )
        ttnn.deallocate(rows)

    fails, checked = [], []

    def check(end, what):
        _set(monkeypatch, CHEAP_CCL="legacy")
        caches["legacy"].write_snapshot(end)
        _set(monkeypatch, CHEAP_CCL="on")
        poison(caches["on"])
        caches["on"].write_snapshot(end)
        ttnn.synchronize_device(mesh)
        cap = snapshot_cap(end, 64)
        want = torch.zeros(SNAP, WIDTH, dtype=torch.bfloat16)
        for p in range(max(0, cap - SNAP), cap):
            want[p % SNAP] = full[p]
        ring = caches["on"].ring
        owners = sorted({str(ring.own[(p // 32) % ring.BLOCKS]) for p in range(max(0, cap - SNAP), cap, 32)})
        for q, got in enumerate(host_batches()):
            for b, side in ((1, "legacy"), (2, "on")):
                if not torch.equal(got[b].view(torch.int16), want.view(torch.int16)):
                    bad = torch.nonzero((got[b] != want).any(dim=-1)).flatten()
                    fails.append(f"{what} end {end} cap {cap}: {side} chip {q} rows {bad[:8].tolist()} differ")
            if not torch.equal(got[0].view(torch.int16), guard[q][0].view(torch.int16)):
                fails.append(f"{what} end {end}: chip {q} guard batch 0 changed")
        checked.append((what, end, cap, owners))

    # (cap 1024: rows [896, 1024) span chips 5 and 6 on 8x4 (160 rows a chip); the next case's carry [864, 992) is
    # still in the ring)
    mixed = 1030
    for start, end, how, what in (
        (0, 100, "rep", "short prompt, cap < 128"),
        (0, chunk, "own", "whole chunk, owned blocks"),
        (chunk, chunk + mixed, "own", "mid-chunk end, two owner chips"),
        (chunk + 992, 2 * chunk + 70, "rep", "follow-up at 32 mod 64, window over two chunks"),
        (2 * chunk + 64, 3 * chunk - 23, "own", "owned blocks, end mid-record"),
        (0, 40, "rep", "cap 0: no position below cap"),
    ):
        write(how, start, end)
        check(end, what)
    print(f"[snapshot send] checked (what, end, cap, owners): {checked}", flush=True)
    assert not fails, "\n".join(fails[:20])
    print("P.65 snapshot send: legacy == on == definition, every chip, every case", flush=True)

    # timing on the steady pattern (the last chip owns the 4 blocks)
    write("own", 0, chunk)
    end = chunk

    def call(side):
        _set(monkeypatch, CHEAP_CCL=side)
        caches[side].write_snapshot(end)

    for side in ("legacy", "on"):
        call(side)
    times = {s: [] for s in ("legacy", "on")}
    for it in range(RUNS + 2):
        for s in times:
            _set(monkeypatch, CHEAP_CCL=s)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            caches[s].write_snapshot(end)
            dt = time.perf_counter() - t0
            ttnn.synchronize_device(mesh)
            if it >= 2:
                times[s].append(dt)
    print(
        "P.65 snapshot host us a call (median): "
        + str({s: round(1e6 * statistics.median(v), 1) for s, v in times.items()}),
        flush=True,
    )
    burst = 40
    walls = {s: [] for s in times}
    for it in range(6):
        for s in walls:
            _set(monkeypatch, CHEAP_CCL=s)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(burst):
                caches[s].write_snapshot(end)
            ttnn.synchronize_device(mesh)
            if it >= 1:
                walls[s].append((time.perf_counter() - t0) / burst)
    print(
        "P.65 snapshot back-to-back wall us a call (median): "
        + str({s: round(1e6 * statistics.median(v), 1) for s, v in walls.items()}),
        flush=True,
    )
    from models.demos.common.bringup.testing import profiler

    dev = {}
    for s in times:
        _set(monkeypatch, CHEAP_CCL=s)
        calls = 15
        profiler.enable(mesh, rt=True)
        try:
            for _ in range(calls):
                caches[s].write_snapshot(end)
            profiler.signpost("end")
            profiler.finish_rt()
            recs = profiler._P["rt_records"]
            tot = sum(max(e[0] for e in chips.values()) for chips in recs.values() if chips)
            dev[s] = (round(tot / 1e3 / calls, 1), round(len(recs) / calls, 1))
        finally:
            profiler.disable()
    print(f"P.65 snapshot device us a call, programs a call: {dev}", flush=True)
    for c in caches.values():
        c.free()
    ttnn.deallocate(slab_t)
