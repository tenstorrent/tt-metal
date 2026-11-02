# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSV41_CMP_FUSE (tt/cmp_fuse.py): bit equality and traced time of the compressor data-movement programs against the composites of SpecCompressedAttention (4x8 mesh, all 32 chips).

compress_shift vs  to_layout(cs) / to_layout(prev) / concat / slice / reshape / to_layout(TILE)
commit_select  vs  gather(cs_block RM) / reshape / to_layout(TILE) / copy   (and mode 0: slice of the last block row)
Env FD_OUT (json of the results)."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import cmp_fuse as CF

OUT = os.environ.get("FD_OUT", "/tmp/fusedec_cmp")
HD = 512
CASES = [(1, 4), (2, 4), (2, 6), (4, 4), (4, 2), (1, 6)]


def _up(md, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def _all(md, t):
    return [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]


def old_shift(cs, prev_cs, U, n):
    T = U * n
    cs3 = ttnn.reshape(ttnn.to_layout(cs, ttnn.ROW_MAJOR_LAYOUT), [U, n, 2 * HD])
    prev = ttnn.reshape(ttnn.to_layout(prev_cs, ttnn.ROW_MAJOR_LAYOUT), [U, 1, 2 * HD])
    prev_all = prev if n == 1 else ttnn.concat([prev, ttnn.slice(cs3, [0, 0, 0], [U, n - 1, 2 * HD])], dim=1)
    return ttnn.to_layout(ttnn.reshape(prev_all, [1, 1, T, 2 * HD]), ttnn.TILE_LAYOUT), cs3


def old_commit(cs3, prev_cs, gidx, U, n):
    if gidx is not None:
        sel = ttnn.gather(cs3, 1, gidx)
    else:
        sel = ttnn.slice(cs3, [0, n - 1, 0], [U, n, 2 * HD])
    ttnn.copy(ttnn.to_layout(ttnn.reshape(sel, [1, 1, U, 2 * HD]), ttnn.TILE_LAYOUT), prev_cs)


def traced_us(md, fn, k=8, reps=20):
    def cap(kk):
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(kk):
            fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        return tid

    def run(tid):
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t) / reps * 1e6

    t1, tk = cap(1), cap(k)
    a, b = run(t1), run(tk)
    return (b - a) / (k - 1)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@torch.no_grad()
def test_cmp_fuse(mesh_device):
    md = mesh_device
    res = {}
    g = torch.Generator().manual_seed(1)
    for U, n in CASES:
        T = U * n
        cs_t = torch.randn(1, 1, T, 2 * HD, generator=g) * 3.0
        pv_t = torch.randn(1, 1, U, 2 * HD, generator=g) * 3.0
        cs, pv = _up(md, cs_t), _up(md, pv_t)
        assert CF.ok(cs, pv, U, n)
        a_old, cs3 = old_shift(cs, pv, U, n)
        a_new = CF.compress_shift(cs, pv, U, n)
        ttnn.synchronize_device(md)
        ao, an = _all(md, a_old), _all(md, a_new)
        eq_shift = all(torch.equal(x[..., :T, :], y[..., :T, :]) for x, y in zip(ao, an))
        # commit: random accepted index per user, uint32 [U,1,1024] rows
        m = torch.randint(0, n, (U,), generator=g)
        gi = m.reshape(U, 1, 1).repeat(1, 1, 2 * HD).to(torch.int32)
        gidx = ttnn.from_torch(
            gi,
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        eq_commit = {}
        for mode, ix in (("gather", gidx), ("last", None)):
            pv_old = _up(md, pv_t.clone())
            pv_new = _up(md, pv_t.clone())
            old_commit(cs3, pv_old, ix, U, n)
            CF.commit_select(cs, pv_new, ix, U, n)
            ttnn.synchronize_device(md)
            po, pn = _all(md, pv_old), _all(md, pv_new)
            eq_commit[mode] = all(torch.equal(x[..., :U, :], y[..., :U, :]) for x, y in zip(po, pn))
            want = torch.stack([cs_t[0, 0, u * n + (int(m[u]) if ix is not None else n - 1)] for u in range(U)])
            eq_commit[mode + "_vs_torch"] = bool(torch.equal(pn[0][0, 0, :U], want))
        # traced time per call (marginal of 8 chained calls)
        us_old_shift = traced_us(md, lambda: old_shift(cs, pv, U, n))
        us_new_shift = traced_us(md, lambda: CF.compress_shift(cs, pv, U, n))
        pv2 = _up(md, pv_t.clone())
        us_old_commit = traced_us(md, lambda: old_commit(cs3, pv2, gidx, U, n))
        us_new_commit = traced_us(md, lambda: CF.commit_select(cs, pv2, gidx, U, n))
        res[f"U{U}_n{n}"] = dict(
            eq_shift=eq_shift,
            eq_commit=eq_commit,
            us_old_shift=us_old_shift,
            us_new_shift=us_new_shift,
            us_old_commit=us_old_commit,
            us_new_commit=us_new_commit,
        )
        print("CMPFUSE " + json.dumps({f"U{U}_n{n}": res[f"U{U}_n{n}"]}), flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    for k, v in res.items():
        assert v["eq_shift"] and all(v["eq_commit"].values()), (k, v)
