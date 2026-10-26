# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/route_table`` vs the regions MoE's composite routing ops (tt/moe.py ``_forward_regions``, REGIONS_RANK
mm; tt-blaze G53F-0195) on the 8x4 mesh at the server's shapes (T 8192, k 8, E 288, 72 local experts): ``pos_tk``,
the masked weights ``w x loc_tk``, ``counts`` and ``offsets``, bitwise on every chip, for two expert ranges (columns 0
and 2) and routing weights that include exact zeros, an fp32 subnormal and a tiny normal at some picks; then both paths
timed (REPS back-to-back calls, one synchronize).

    pytest models/demos/glm53_flash/tests/custom_ops/test_route_table.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.route_table.op import RouteTable
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.moe import _compute

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("RT_T", "8192"))
K, E, E_LOC, TILE = 8, 288, 72, 32
REPS = int(os.environ.get("RT_REPS", "10"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("col", [0, 2])
def test_route_table(mesh_device, col):
    g = torch.Generator().manual_seed(5 + col)
    idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])  # distinct experts per token
    w = torch.rand(T, K, generator=g) * 0.5 + 0.01
    w[torch.rand(T, K, generator=g) < 0.002] = 0.0
    w[3, 1], w[5, 2], w[7, 0] = 1e-40, 1e-37, 0.0  # an fp32 subnormal, a tiny normal, a zero
    lo = col * E_LOC
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    up = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, dtype=dt, layout=lay, device=mesh_device, mesh_mapper=rep
    )
    idx_t = up(idx.reshape(1, 1, T, K).float(), ttnn.float32)
    w_t = up(w.reshape(1, 1, T, K), ttnn.float32)
    mask = torch.zeros(1, 1, 1, E)
    mask[..., lo : lo + E_LOC] = 1.0
    mask_t = up(mask, ttnn.float32)
    table = torch.full((1, E), -1, dtype=torch.int32)
    table[0, lo : lo + E_LOC] = 0
    table_t = up(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    tril = up(torch.tril(torch.ones(T, T, dtype=torch.bfloat16)).reshape(1, 1, T, T), ttnn.bfloat16)
    zeros = up(torch.zeros(1, 1, T, E, dtype=torch.bfloat16), ttnn.bfloat16)
    rank_cfg = _compute(mesh_device, ttnn.MathFidelity.HiFi4)
    max_buf = -(-(T * K + E_LOC * TILE) // TILE) * TILE

    def dense_of():  # the router's dense matrix (TtRouter._forward_topk): bf16(w) scattered at the picks
        w16 = ttnn.typecast(w_t, ttnn.bfloat16)
        iu = ttnn.typecast(idx_t, ttnn.uint32)
        d = ttnn.scatter(zeros, -1, iu, w16)
        ttnn.deallocate(w16)
        ttnn.deallocate(iu)
        return d

    dense_w = dense_of()

    def composite():
        onehot = ttnn.multiply(ttnn.typecast(ttnn.ne(dense_w, 0.0), ttnn.float32), mask_t)
        oh16 = ttnn.typecast(onehot, ttnn.bfloat16)
        rank = ttnn.matmul(tril, oh16, dtype=ttnn.float32, compute_kernel_config=rank_cfg)
        ttnn.deallocate(oh16)
        counts_f = ttnn.slice(rank, [0, 0, T - 1, 0], [1, 1, T, E])
        aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(counts_f, TILE - 1), 1.0 / TILE)), float(TILE))
        incl = ttnn.cumsum(aligned, dim=-1)
        offsets_f = ttnn.subtract(incl, aligned)
        pos_te = ttnn.add(
            ttnn.multiply(ttnn.add(ttnn.subtract(rank, 1.0), offsets_f), onehot),
            ttnn.multiply(ttnn.rsub(onehot, 1.0), float(max_buf - 1)),
        )
        for t in (rank, aligned, incl):
            ttnn.deallocate(t)
        idx_u32 = ttnn.typecast(idx_t, ttnn.uint32)
        pos_tk = ttnn.gather(pos_te, -1, idx_u32)
        loc_tk = ttnn.gather(onehot, -1, idx_u32)
        for t in (pos_te, onehot, idx_u32):
            ttnn.deallocate(t)
        counts = ttnn.reshape(ttnn.to_layout(ttnn.typecast(counts_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
        offsets = ttnn.reshape(ttnn.to_layout(ttnn.typecast(offsets_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
        ttnn.deallocate(counts_f)
        ttnn.deallocate(offsets_f)
        w_loc = ttnn.multiply(w_t, loc_tk)
        ttnn.deallocate(loc_tk)
        return pos_tk, w_loc, counts, offsets

    rt = RouteTable(mesh_device, E_LOC)
    custom = lambda: rt(idx_t, w_t, table_t, max_buf)

    a, b = composite(), custom()
    names = ("pos_tk", "w_loc", "counts", "offsets")
    eq = {n: 0 for n in names}
    first = {}
    for n, ta, tb in zip(names, a, b):
        for da, db in zip(ttnn.get_device_tensors(ta), ttnn.get_device_tensors(tb)):
            xa, xb = ttnn.to_torch(da), ttnn.to_torch(db)
            xa = xa.view(torch.int32) if xa.dtype != torch.int32 else xa
            xb = xb.view(torch.int32) if xb.dtype != torch.int32 else xb
            ok = xa.shape == xb.shape and torch.equal(xa, xb)
            eq[n] += int(ok)
            if not ok and n not in first:
                d = (xa != xb).nonzero()[:4].tolist() if xa.shape == xb.shape else (tuple(xa.shape), tuple(xb.shape))
                first[n] = d
    for t in (*a, *b):
        ttnn.deallocate(t)
    # host reference of the slots (an independent check of both)
    loc = (w > 0) & (w > 1e-38) & (idx >= lo) & (idx < lo + E_LOC)
    cnt = torch.zeros(E, dtype=torch.int64)
    cnt.index_add_(0, idx[loc], torch.ones(int(loc.sum()), dtype=torch.int64))
    logger.info(f"[route-table] col {col}: local picks {int(loc.sum())}, max count {int(cnt.max())}")

    def timed(fn):
        for _ in range(2):
            for t in fn():
                ttnn.deallocate(t)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            for t in fn():
                ttnn.deallocate(t)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / REPS * 1e3

    t_c, t_r = timed(composite), timed(custom)
    chips = len(ttnn.get_device_tensors(idx_t))
    logger.info(
        f"[route-table] col {col} T={T} k={K} E={E}: bitwise chips {eq} of {chips} (first diffs {first}); "
        f"composite {t_c:.3f} ms, route table {t_r:.3f} ms per sub-chunk"
    )
    assert all(v == chips for v in eq.values()), (eq, first)
