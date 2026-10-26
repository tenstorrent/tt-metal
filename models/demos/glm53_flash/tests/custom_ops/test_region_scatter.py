# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/region_scatter`` + the fused combine on the fp32 TILE slots (tt-blaze G53F-0193), on the 8x4 mesh at
the regions path's shapes (T 8192, k 8, H 4096, ~25 % local picks):

* the region buffer: RegionScatter(x, pos_tk) vs today's slot copy (typecast + untilize) + ttnn.scatter +
  ttnn.embedding, compared on every local slot (the only rows the expert kernel's used outputs depend on);
* GatherCombineReduce(table, pos_tk) vs GatherCombineReduce(table, pos_rm): bitwise.

Both on every chip; both paths timed (5 warm calls).

    pytest models/demos/glm53_flash/tests/custom_ops/test_region_scatter.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.gather_combine_reduce.op import GatherCombineReduce
from models.demos.glm53_flash.custom_ops.region_scatter.op import RegionScatter
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("RS_T", "8192"))
K = 8
H = 4096
REPS = int(os.environ.get("RS_REPS", "5"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_region_scatter(mesh_device):
    g = torch.Generator().manual_seed(1)
    local = torch.rand(T, K, generator=g) < 0.25
    n_local = int(local.sum())
    R = T * K + 72 * 32  # the full regions buffer
    pos = torch.full((T, K), R - 1, dtype=torch.int64)
    pos[local] = torch.randperm(R - 1, generator=g)[:n_local]
    x_h = torch.randn(T, H, generator=g).to(torch.bfloat16)
    w_h = (torch.rand(T, K, generator=g) * 0.5 + 0.01) * local
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    up = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh_device, mesh_mapper=rep)
    x_rm = up(x_h, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    pos_tk = up(pos.reshape(1, 1, T, K).float(), ttnn.float32, ttnn.TILE_LAYOUT)
    iota = up(
        torch.arange(T, dtype=torch.int32).repeat_interleave(K).reshape(1, T * K), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    )
    zeros = up(torch.zeros(1, R, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    w_rm = up(w_h.to(torch.bfloat16).reshape(1, 1, T, K, 1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    rs, gcr = RegionScatter(mesh_device), GatherCombineReduce(mesh_device)

    def pos_copy():
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(pos_tk, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, T * K])

    def ref_buf():
        prm = pos_copy()
        slot_src = ttnn.scatter(zeros, -1, prm, iota)
        buf = ttnn.embedding(slot_src, x_rm, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(slot_src)
        ttnn.deallocate(prm)
        return buf

    def new_buf():
        return rs(x_rm, pos_tk, R)

    # 1. the region buffer, on the local slots
    a, b = ref_buf(), new_buf()
    slots = pos[local].unique()
    eq_buf = []
    for da, db in zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)):
        ra, rb = ttnn.to_torch(da).reshape(R, H), ttnn.to_torch(db).reshape(R, H)
        eq_buf.append(torch.equal(ra[slots], rb[slots]))
    # the expected content of a local slot: its token's row
    tok = torch.arange(T).unsqueeze(1).expand(T, K)[local]
    ra0 = ttnn.to_torch(ttnn.get_device_tensors(b)[0]).reshape(R, H)
    exp_ok = torch.equal(ra0[pos[local]], x_h[tok])

    # 2. the combine on the TILE slots vs the uint32 copy
    table = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT)  # any [R, H] bf16 table with finite rows
    prm = pos_copy()
    y1, y2 = gcr(table, prm, w_rm), gcr(table, pos_tk, w_rm)
    eq_comb = [
        torch.equal(ttnn.to_torch(p), ttnn.to_torch(q))
        for p, q in zip(ttnn.get_device_tensors(y1), ttnn.get_device_tensors(y2))
    ]
    for t in (a, y1, y2):
        ttnn.deallocate(t)

    def timed(fn):
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / REPS * 1e3

    t_ref, t_new = timed(ref_buf), timed(new_buf)
    t_c1, t_c2 = timed(lambda: gcr(table, pos_copy(), w_rm)), timed(lambda: gcr(table, pos_tk, w_rm))
    logger.info(
        f"[region-scatter] T={T} k={K} R={R} local={n_local}: buffer equal on the local slots {sum(eq_buf)}/{len(eq_buf)} "
        f"chips, rows == x[token] {exp_ok}; combine (TILE slots) == combine (uint32 copy) {sum(eq_comb)}/{len(eq_comb)}"
    )
    logger.info(
        f"[region-scatter] buffer: slot copy + scatter + embedding {t_ref:.3f} ms, region scatter {t_new:.3f} ms; "
        f"combine: with the slot copy {t_c1:.3f} ms, on the TILE slots {t_c2:.3f} ms"
    )
    assert all(eq_buf) and exp_ok and all(eq_comb), (eq_buf, exp_ok, eq_comb)
