# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.64 (ATTN_FUSE=on): the fused window write (tt/window_send.WindowSend.fused) against the legacy ops it replaces,
on random rows, every chip, byte for byte; and the host us a call of both.

Per case (ring pieces as tt/attention.WindowRing.plan makes them: none, one whole, two, a wrap, four): every chip its
own random bf16 rows [1, 1, q_rows, 512] (q_rows = chunk / chips), LAYERS layers each with its own region
[1, 1, 128 + q_rows, 512], kept tail [1, 1, 128, 512] and ring [1, 1, 192, 512] (random starting bytes).
legacy = ttnn.bringup.slice_write of the rows into region row 128, per ring piece ttnn.slice (unless the whole rows)
+ ttnn.bringup.slice_write, then WindowSend (the send of the region's last 128 rows); on = WindowSend.fused. The
two sides start from the same bytes; every chip's region, tail and ring must be equal after each call, and the region
/ tail must hold what the send defines (chip q + 1's carry = chip q's last 128 rows; chip 0's = the last chip's old
tail; the last chip's new tail = its own last 128 rows).

Host us a call (PROBE_RUNS calls a side, alternating, median): legacy group (its 3-8 launches) vs the fused call, and
the fused call with WW_CORES copy cores (default 1,2,4).

    scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_window_write_fused.py -s
"""

from __future__ import annotations

import os
import statistics
import time

import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
LAYERS = 3
RUNS = int(os.environ.get("PROBE_RUNS", "30"))
CORES = [int(x) for x in os.environ.get("WW_CORES", "1,2,4").split(",")]
DRAM = ttnn.DRAM_MEMORY_CONFIG
RING = 192
W = 128
# (row0 of the rows, ring slot, rows) pieces; q_rows-relative cases are filled in per mesh
CASES = {
    "none": lambda q: [],
    "two": lambda q: [(q - 32, 64, 32), (0, 96, 96)] if q >= 96 else [(0, 64, 32)],
    "whole": lambda q: [(0, 0, min(q, RING))] if q <= RING else [(q - RING, 0, RING)],
    "wrap": lambda q: [(q - 64, 160, 32), (q - 32, 0, 32)],
    "four": lambda q: [(0, 128, 32), (32, 160, 32), (64, 0, 32), (96, 32, 32)],
}


def _rand(mesh, rows, gen):
    R, C = tuple(mesh.shape)
    t = torch.randn(R, C, rows, 512, generator=gen).to(torch.bfloat16)
    mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(0, 1))
    return t, ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=mapper
    )


def _host(t):
    return torch.stack([ttnn.to_torch(x).reshape(-1, 512) for x in ttnn.get_device_tensors(t)])


def _legacy(send, rows, region, tail, ring, pieces, q_rows):
    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

    ttnn.bringup.slice_write(rows, region, [0, 0, W, 0], [1, 1, W + q_rows, 512], [1, 1, 1, 1], max_grid=max_grid())
    grid = ttnn.CoreCoord(8, 1)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    for r, slot, take in pieces:
        whole = r == 0 and take == q_rows
        piece = (
            rows
            if whole
            else ttnn.slice(rows, [0, 0, r, 0], [1, 1, r + take, 512], memory_config=DRAM, sub_core_grids=cores)
        )
        ttnn.bringup.slice_write(piece, ring, [0, 0, slot, 0], [1, 1, slot + take, 512], [1, 1, 1, 1], max_grid=grid)
        if not whole:
            ttnn.deallocate(piece)
    send(region, tail)


@mesh_parametrize(timeout_s=1800)
def test_window_write_fused(mesh_device):
    from models.demos.deepseek_v41_flash_d_p.tt.window_send import WindowSend

    mesh = mesh_device
    n = mesh.get_num_devices()
    chunk = int(S.get("target.chunk"))
    q_rows = chunk // n
    assert q_rows >= W and q_rows % 32 == 0, q_rows
    gen = torch.Generator().manual_seed(64)
    sends = {c: WindowSend(mesh, W + q_rows, 1024, W, copy_cores=c) for c in CORES}
    legacy_send = sends[CORES[0]]
    for name, mk in CASES.items():
        pieces = mk(q_rows)
        for c in CORES:
            # each layer's buffers on both sides from the same random bytes
            bufs = []
            for _ in range(LAYERS):
                side = []
                hs = [_rand(mesh, k, gen)[0] for k in (W + q_rows, W, RING)]
                for _s in range(2):
                    side.append(
                        [
                            ttnn.from_torch(
                                h,
                                dtype=ttnn.bfloat16,
                                layout=ttnn.ROW_MAJOR_LAYOUT,
                                device=mesh,
                                memory_config=DRAM,
                                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)),
                            )
                            for h in hs
                        ]
                    )
                bufs.append((hs, side))
            for layer, (hs, (lg, fu)) in enumerate(bufs):
                rows_h, rows = _rand(mesh, q_rows, gen)
                _legacy(legacy_send, rows, *lg, pieces, q_rows)
                sends[c].fused(fu[0], fu[1], rows, fu[2], pieces)
                ttnn.synchronize_device(mesh)
                got = [[_host(t) for t in s] for s in (lg, fu)]
                for k, what in enumerate(("region", "tail", "ring")):
                    assert torch.equal(got[0][k], got[1][k]), f"{name} cores {c} layer {layer}: {what} differs"
                # the send's definition, on the legacy side's bytes
                own = rows_h.reshape(n, q_rows, 512)
                region, tail = got[0][0], got[0][1]
                old_tail = hs[1].reshape(n, W, 512)
                for q in range(n):
                    carry = old_tail[n - 1] if q == 0 else own[q - 1, q_rows - W :]
                    assert torch.equal(region[q, :W], carry), f"{name}: chip {q} carry"
                    assert torch.equal(region[q, W:], own[q]), f"{name}: chip {q} own rows"
                assert torch.equal(tail[n - 1], own[n - 1, q_rows - W :]), f"{name}: last chip's kept tail"
                ring_ref = hs[2].reshape(n, RING, 512).clone()
                for r, slot, take in pieces:
                    ring_ref[:, slot : slot + take] = own[:, r : r + take]
                assert torch.equal(got[0][2], ring_ref), f"{name}: ring"
                ttnn.deallocate(rows)
            for hs, side in bufs:
                for s in side:
                    for t in s:
                        ttnn.deallocate(t)
        print(f"P.64 window write {name}: pieces {pieces}: legacy == fused (cores {CORES}), every chip", flush=True)

    # host us a call (buffers reused; the rows allocated per call as in the model)
    pieces = CASES["two"](q_rows)
    reg = [
        ttnn.zeros(
            [1, 1, W + q_rows, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM
        )
        for _ in range(LAYERS)
    ]
    tl = [
        ttnn.zeros([1, 1, W, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM)
        for _ in range(LAYERS)
    ]
    rg = [
        ttnn.zeros(
            [1, 1, RING, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM
        )
        for _ in range(LAYERS)
    ]
    rows = _rand(mesh, q_rows, gen)[1]
    sides = ["legacy"] + [f"fused{c}" for c in CORES]
    times = {s: [] for s in sides}
    for it in range(RUNS + 2):
        for s in sides:
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for layer in range(LAYERS):
                if s == "legacy":
                    _legacy(legacy_send, rows, reg[layer], tl[layer], rg[layer], pieces, q_rows)
                else:
                    sends[int(s[5:])].fused(reg[layer], tl[layer], rows, rg[layer], pieces)
            dt = (time.perf_counter() - t0) / LAYERS
            ttnn.synchronize_device(mesh)
            if it >= 2:
                times[s].append(dt)
    print(
        "P.64 window write host us a layer (median): "
        + str({s: round(1e6 * statistics.median(v), 1) for s, v in times.items()}),
        flush=True,
    )
    # back to back: BURST layers' calls then one sync, wall us a call (the device's time when it is the slower side)
    burst = 40
    walls = {s: [] for s in sides}
    for it in range(6):
        for s in sides:
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for j in range(burst):
                layer = j % LAYERS
                if s == "legacy":
                    _legacy(legacy_send, rows, reg[layer], tl[layer], rg[layer], pieces, q_rows)
                else:
                    sends[int(s[5:])].fused(reg[layer], tl[layer], rows, rg[layer], pieces)
            ttnn.synchronize_device(mesh)
            if it >= 1:
                walls[s].append((time.perf_counter() - t0) / burst)
    print(
        "P.64 window write back-to-back wall us a layer (median): "
        + str({s: round(1e6 * statistics.median(v), 1) for s, v in walls.items()}),
        flush=True,
    )
    # device us a layer: the real-time profiler (each program's slowest chip, summed over the side's programs)
    from models.demos.common.bringup.testing import profiler

    dev = {}
    for s in sides:
        calls = 5 * LAYERS
        profiler.enable(mesh, rt=True)
        try:
            for j in range(calls):
                layer = j % LAYERS
                if s == "legacy":
                    _legacy(legacy_send, rows, reg[layer], tl[layer], rg[layer], pieces, q_rows)
                else:
                    sends[int(s[5:])].fused(reg[layer], tl[layer], rows, rg[layer], pieces)
            profiler.signpost("end")
            profiler.finish_rt()
            recs = profiler._P["rt_records"]
            tot = sum(max(e[0] for e in chips.values()) for chips in recs.values() if chips)
            dev[s] = (round(tot / 1e3 / calls, 1), round(len(recs) / calls, 1))
        finally:
            profiler.disable()
    print(f"P.64 window write device us a layer, programs a layer: {dev}", flush=True)
