# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill device MoE routing (tt/fused/pf_route.py) vs the host routing of prefill_model (_route_unified):
offsets, counts, back index and the row index of every routed row equal, for uniform and skewed expert ids,
full and partial chunks; plus the device time at T = 8192."""

import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_route import PfRoute

NE, K = 128, 8


def urows(t):  # prefill_model.Inkling2Prefill._urows
    return -(-(t * K + NE * 31) // 32) * 32 + 32


def host_route(ids, n, t):  # prefill_model.Inkling2Prefill._route_unified on ids[:n]
    nrows = urows(t)
    flat = ids[:n].reshape(-1).astype(np.int64)
    counts = np.bincount(flat, minlength=NE)
    cap = -(-counts // 32) * 32
    off = np.concatenate([[0], np.cumsum(cap)[:-1]])
    order = np.argsort(flat, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size) - starts[flat[order]]
    dest = off[flat] + rank
    xri = np.full(nrows, t, dtype=np.int64)
    xri[dest] = np.arange(flat.size) // K
    back = np.full((K, t), nrows - 32, dtype=np.int64)
    back[:, :n] = dest.reshape(n, K).T
    return off, counts, xri, back.reshape(-1), dest


def host_slots(ids, n, t, gm, nslot):  # prefill_model.Inkling2Prefill._route_host, sparse slots
    flat = ids[:n].reshape(-1).astype(np.int64)
    counts = np.bincount(flat, minlength=NE)
    per = -(-counts // gm)
    used = int(per.sum())
    slot0 = np.concatenate([[0], np.cumsum(per)[:-1]])
    eid = np.empty(nslot, dtype=np.int64)
    eid[:used] = np.repeat(np.arange(NE), per)
    eid[used:] = eid[used - 1]
    order = np.argsort(flat, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size) - starts[flat[order]]
    dest = slot0[flat] * gm + rank
    gidx = np.zeros(nslot * gm, dtype=np.int64)
    gidx[dest] = np.arange(flat.size) // K
    back = np.zeros((K, t), dtype=np.int64)
    back[:, :n] = dest.reshape(n, K).T
    return eid, gidx, back.reshape(-1), dest


def make_ids(t, skew, g):
    if skew:  # most pairs on 3 experts (one expert can take most tokens of a chunk)
        p = torch.full((NE,), 0.1 / (NE - 3))
        p[[5, 77, 127]] = 0.3
        return torch.stack([torch.multinomial(p, K, generator=g) for _ in range(t)]).numpy()
    return torch.stack([torch.randperm(NE, generator=g)[:K] for _ in range(t)]).numpy()


@pytest.mark.parametrize("t,n,skew", [(256, 256, False), (256, 200, True), (8192, 8192, False), (8192, 8100, True)])
def test_pf_route(device, t, n, skew):
    dev = device
    g = torch.Generator().manual_seed(t + n)
    ids = make_ids(t, skew, g)
    rm, dram = ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
    put = lambda a, dt, lay=rm: ttnn.from_torch(a, dtype=dt, layout=lay, device=dev, memory_config=dram)  # noqa: E731
    idx = put(torch.from_numpy(ids.astype(np.int32)).view(1, 1, t, K), ttnn.uint16, ttnn.TILE_LAYOUT)
    nrows = urows(t)
    nreal = put(torch.tensor([[n] + [0] * 15], dtype=torch.int32), ttnn.uint32)
    z = lambda m: put(torch.zeros(1, m, dtype=torch.int32), ttnn.uint32)  # noqa: E731
    off, cnt, xri, back = z(NE), z(NE), z(nrows), z(K * t)
    mod = PfRoute(dev)
    hist = mod.hist_buffer(t, NE)
    mod(idx, nreal, hist, off, cnt, xri, back, nrows - 32, K)
    r_off, r_cnt, r_xri, r_back, dest = host_route(ids, n, t)
    d = lambda x: ttnn.to_torch(x).reshape(-1).to(torch.int64).numpy()  # noqa: E731
    d_off, d_cnt, d_xri, d_back = d(off), d(cnt), d(xri), d(back)
    ok = dict(off=np.array_equal(d_off, r_off), cnt=np.array_equal(d_cnt, r_cnt), back=np.array_equal(d_back, r_back),
              xri=np.array_equal(d_xri[dest], r_xri[dest]))  # fmt: skip
    print(f"PF_ROUTE t={t} n={n} skew={skew} {ok} max_cnt={int(r_cnt.max())}")
    assert all(ok.values()), ok
    if t == 8192 and not skew:
        mod(idx, nreal, hist, off, cnt, xri, back, nrows - 32, K)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(10):
            mod(idx, nreal, hist, off, cnt, xri, back, nrows - 32, K)
        ttnn.synchronize_device(dev)
        print(f"PF_ROUTE_MS t={t} {(time.perf_counter() - t0) / 10 * 1e3:.3f}")


@pytest.mark.parametrize("t,n,skew", [(256, 200, True), (8192, 8192, False), (8192, 8100, True)])
def test_pf_route_slots(device, t, n, skew):
    """Sparse-matmul slots (64 rows per slot, expert per slot) vs the host routing."""
    dev, gm = device, 64
    nslot = -(-(-(-t * K // gm) + NE) // 32) * 32  # prefill_model._nslot
    g = torch.Generator().manual_seed(t + n + 1)
    ids = make_ids(t, skew, g)
    rm, dram = ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
    put = lambda a, dt, lay=rm: ttnn.from_torch(a, dtype=dt, layout=lay, device=dev, memory_config=dram)  # noqa: E731
    idx = put(torch.from_numpy(ids.astype(np.int32)).view(1, 1, t, K), ttnn.uint16, ttnn.TILE_LAYOUT)
    nreal = put(torch.tensor([[n] + [0] * 15], dtype=torch.int32), ttnn.uint32)
    gi = put(torch.zeros(1, nslot * gm, dtype=torch.int32), ttnn.uint32)
    back = put(torch.zeros(1, K * t, dtype=torch.int32), ttnn.uint32)
    eid = put(torch.zeros(1, 1, 1, nslot, dtype=torch.int32), ttnn.uint16)
    mod = PfRoute(dev)
    mod(idx, nreal, mod.hist_buffer(t, NE), None, None, gi, back, 0, K, align=gm, eid=eid)
    r_eid, r_gi, r_back, dest = host_slots(ids, n, t, gm, nslot)
    d = lambda x: ttnn.to_torch(x).reshape(-1).to(torch.int64).numpy()  # noqa: E731
    ok = dict(eid=np.array_equal(d(eid), r_eid), back=np.array_equal(d(back), r_back),
              gidx=np.array_equal(d(gi)[dest], r_gi[dest]))  # fmt: skip
    print(f"PF_ROUTE_SLOTS t={t} n={n} skew={skew} {ok}")
    assert all(ok.values()), ok
