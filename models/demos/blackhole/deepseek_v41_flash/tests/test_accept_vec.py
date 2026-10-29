# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SpecDecoder accept rule (leading drafts d_{j+1} == argmax of row j): the per-column op chain of ``_accept_commit_draft`` vs the vectorised eq / cumprod / sum
(DSV41_SPEC_ACCEPT_VEC=1), equal for every accept count, timed (traced)."""

import time

import pytest
import torch

import ttnn


def _old(a, x, U, n):
    rm = ttnn.ROW_MAJOR_LAYOUT
    a2 = ttnn.typecast(ttnn.reshape(a, [U, n]), ttnn.float32)
    x2 = ttnn.typecast(ttnn.reshape(x, [U, n]), ttnn.float32)
    col = lambda t, j: ttnn.to_layout(ttnn.slice(t, [0, j], [U, j + 1]), ttnn.TILE_LAYOUT)
    c, mcount = None, None
    for j in range(n - 1):
        e = ttnn.eq(col(a2, j), col(x2, j + 1))
        c = e if c is None else ttnn.multiply(c, e)
        mcount = c if mcount is None else ttnn.add(mcount, c)
    return mcount


def _new(a, x, U, n):
    a2 = ttnn.typecast(ttnn.reshape(a, [U, n]), ttnn.float32)
    x2 = ttnn.typecast(ttnn.reshape(x, [U, n]), ttnn.float32)
    e = ttnn.eq(
        ttnn.to_layout(ttnn.slice(a2, [0, 0], [U, n - 1]), ttnn.TILE_LAYOUT),
        ttnn.to_layout(ttnn.slice(x2, [0, 1], [U, n]), ttnn.TILE_LAYOUT),
    )
    return ttnn.sum(ttnn.cumprod(e, dim=1), dim=1, keepdim=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 20_000_000}, id="plain")],
    indirect=True,
)
@pytest.mark.parametrize("U,n", [(1, 4), (1, 6), (2, 6), (4, 6)])
@torch.no_grad()
def test_accept_vec(mesh_device, U, n):
    md = mesh_device
    torch.manual_seed(U * 10 + n)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t.reshape(-1, 1).to(torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    for trial in range(8):
        # every accept count 0..n-1 occurs: user u accepts (u + trial) % n leading drafts
        a = torch.randint(0, 1000, (U, n))
        x = torch.randint(0, 1000, (U, n))
        for u in range(U):
            k = (u + trial) % n  # user u accepts k leading drafts
            for j in range(n - 1):
                x[u, j + 1] = a[u, j] if j < k else (a[u, j] + 1)
        if U > 1 and n > 3:
            x[1, n - 1] = a[1, n - 2]  # a late match behind a mismatch must not count
        want = torch.zeros(U)
        for u in range(U):
            k = 0
            while k < n - 1 and x[u, k + 1] == a[u, k]:
                k += 1
            want[u] = k
        a_d, x_d = up(a), up(x)
        for name, fn in (("old", _old), ("new", _new)):
            m = ttnn.to_torch(ttnn.get_device_tensors(fn(a_d, x_d, U, n))[0]).float().reshape(U, -1)[:, 0]
            assert torch.equal(m, want), (name, trial, m, want)
    for name, fn in (("old", _old), ("new", _new)):
        fn(a_d, x_d, U, n)
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(20):
            fn(a_d, x_d, U, n)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        print(f"ACCEPT_VEC U={U} n={n} {name} {(time.perf_counter() - t0) / 100 * 1e6:.1f} us per call", flush=True)
        ttnn.release_trace(md, tid)
