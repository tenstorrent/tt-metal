# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-resident Engram tables (rows sharded over 32 chips): allocation, shard load, gather+dequant kernel and the
cross-chip combine, bit-checked against the host rows (_MappedTable.rows) on REAL checkpoint rows.
Env: DSV41_CHIPS=0,31 (chips whose shard is really loaded; ids are drawn only from those chips' ranges)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.engram import _MappedTable
from models.demos.blackhole.deepseek_v41_flash.tt.engram_table import (
    DSV41EngramGather,
    K,
    alloc_table,
    combine,
    load_table,
    rows_per_chip,
)
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

NUM_ROWS = (384_006_168, 384_016_682)
LAYERS = (1, 14)


def replay_ms(md, fn, n=30):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    dt = (time.perf_counter() - t) / n * 1e3
    ttnn.release_trace(md, tid)
    return dt


def chain_us(md, op):
    def many(n):
        def f():
            for _ in range(n):
                op()

        return f

    return (replay_ms(md, many(41)) - replay_ms(md, many(1))) / 40 * 1e3


def norm0(x):  # -0.0 -> +0.0 on the int16 view (the sum of -0 and +0 partials is +0)
    v = x.view(torch.int16).clone()
    v[v == -32768] = 0
    return v


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_engram_table_device(mesh_device):
    md = mesh_device
    n = md.get_num_devices()
    spec = os.environ.get("DSV41_CHIPS", "0,31")
    chips = list(range(32)) if spec == "all" else [int(c) for c in spec.split(",")]
    sh = _Shards()
    host = [_MappedTable(sh, lid) for lid in LAYERS]
    assert tuple(h.w.shape[0] for h in host) == NUM_ROWS
    rpcs = [rows_per_chip(nr, n) for nr in NUM_ROWS]
    print(f"TABLE rows_per_chip={rpcs}", flush=True)
    print("DRAM before:", ttnn.get_memory_view(md, ttnn.BufferType.DRAM), flush=True)
    tables = []
    for l, rpc in enumerate(rpcs):
        rows, scales = alloc_table(md, rpc)
        tables.append((rows, scales, NUM_ROWS[l]))
    ttnn.synchronize_device(md)
    mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    print("DRAM after alloc of both layers:", mv, flush=True)
    import resource

    for l in range(2):
        th, tw = load_table(md, tables[l][0], tables[l][1], host[l], rpcs[l], chips)
        print(
            f"LOAD layer {LAYERS[l]} chips {spec}: host-build {th:.1f} s, device-write {tw:.1f} s for {len(chips)} chip shards "
            f"({len(chips) * rpcs[l] * 264 / 1e9:.1f} GB real) maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB",
            flush=True,
        )

    # ids: random global rows inside the loaded chips' ranges (+ range boundaries)
    g = torch.Generator().manual_seed(7)
    U = 16
    ids = torch.zeros((U, 64), dtype=torch.int64)
    for l in range(2):
        rpc, nr = rpcs[l], NUM_ROWS[l]
        lo_hi = [(c * rpc, min(nr, (c + 1) * rpc)) for c in chips]
        pick = torch.randint(0, len(chips), (U, K), generator=g)
        for ci, (a, b) in enumerate(lo_hi):
            m = pick == ci
            ids[:, l * K : (l + 1) * K][m] = torch.randint(a, b, (int(m.sum()),), generator=g)
        ids[0, l * K], ids[0, l * K + 1] = lo_hi[0][0], lo_hi[-1][1] - 1  # first / last row of loaded range
        ids[1, l * K] = lo_hi[0][1] - 1
    gather = DSV41EngramGather(md, tables, users=U)
    ids_dev = gather.upload_ids(ids)
    out = gather(ids_dev)
    ttnn.synchronize_device(md)

    # expected rows in kernel row order j = ((g*2+layer)*4 + tl)*K + k  with user u = g*4+tl
    exp = torch.zeros(U * 2 * K, 256, dtype=torch.bfloat16)
    for j in range(U * 2 * K):
        gg, rem = divmod(j, 4 * 2 * K)
        layer, rem = divmod(rem, 4 * K)
        tl, k = divmod(rem, K)
        u = gg * 4 + tl
        exp[j] = host[layer].rows(ids[u, layer * K + k].reshape(1))[0]
    # (1) per-chip gather output: rows in this chip's range equal the host rows, all others exactly zero
    per_chip = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).reshape(n, U * 2 * K, 256)
    bad = 0
    for c in range(n):
        own = torch.zeros(U * 2 * K, dtype=torch.bool)
        for j in range(U * 2 * K):
            gg, rem = divmod(j, 4 * 2 * K)
            layer, rem = divmod(rem, 4 * K)
            tl, k = divmod(rem, K)
            gid = int(ids[gg * 4 + tl, layer * K + k])
            own[j] = gid // rpcs[layer] == c
        want = torch.where(own[:, None], exp, torch.zeros_like(exp))
        d = per_chip[c].view(torch.int16) != want.view(torch.int16)
        bad += int(d.sum())
        if d.any():
            rr = d.any(1).nonzero().flatten().tolist()
            print(f"  chip {c}: {len(rr)} bad rows, e.g. {rr[:6]}; own rows {int(own.sum())}", flush=True)
            r0 = rr[0]
            cols = d[r0].nonzero().flatten().tolist()
            print(
                f"    row {r0}: {len(cols)} bad cols {cols[:8]}; got {per_chip[c][r0].view(torch.int16)[cols[:4]].tolist()} want {want[r0].view(torch.int16)[cols[:4]].tolist()} own={bool(own[r0])}",
                flush=True,
            )
    print(
        f"GATHER exact check: {n} chips x {U * 2 * K} rows, mismatching bf16 elements (bit compare) = {bad}", flush=True
    )
    assert bad == 0
    t_g = chain_us(md, lambda: gather(ids_dev))
    print(f"GATHER_TIME per op in long trace: {t_g:.1f} us", flush=True)

    # (2) combine
    red = combine(out, md)
    ttnn.synchronize_device(md)
    R = ttnn.get_device_tensors(red)
    rows_g = U // 4 * 0 + (U * 2 * K) // 4
    mism = 0
    for r in range(4):
        want = norm0(exp[r * rows_g : (r + 1) * rows_g])
        for c in range(8):
            got = norm0(ttnn.to_torch(R[r * 8 + c]).reshape(rows_g, 256))
            mism += int((got != want).sum())
    print(f"COMBINE exact check (all 32 chips vs host rows, -0==+0): mismatching elements = {mism}", flush=True)
    assert mism == 0
    t_c = chain_us(md, lambda: combine(out, md))
    print(f"COMBINE_TIME per op in long trace (reduce_scatter axis0 + all_reduce axis1): {t_c:.1f} us", flush=True)
