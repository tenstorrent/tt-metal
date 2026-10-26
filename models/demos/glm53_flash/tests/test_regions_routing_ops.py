# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time of the regions MoE's routing-table ops (tt/moe.py ``_forward_regions``, REGIONS_RANK=mm) on the 8x4
mesh at the server's shapes (T 8192 per sub-chunk, k 8, E 288, 72 local experts per chip): the ~30 small ttnn ops
that turn the router's output into the expert kernel's counts / offsets, the picks' region slots and the combine
weights (tt-blaze G53F-0195: is fusing them worth a kernel?). Each group is timed as a sequence of REPS calls with one
synchronize at the end (device-bound back-to-back execution, as in the real prefill), after 2 warm-up calls.

    pytest models/demos/glm53_flash/tests/test_regions_routing_ops.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.moe import _compute

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("RR_T", "8192"))
K, E, E_LOC, H, TILE = 8, 288, 72, 4096, 32
REPS = int(os.environ.get("RR_REPS", "10"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_regions_routing_ops(mesh_device):
    g = torch.Generator().manual_seed(2)
    idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])  # [T, k] distinct experts per token
    w = torch.rand(T, K, generator=g) * 0.5 + 0.01
    dense = torch.zeros(T, E)
    dense.scatter_(1, idx, w)
    mask = torch.zeros(1, 1, 1, E)
    mask[..., :E_LOC] = 1.0  # column 0's experts
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    up = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, dtype=dt, layout=lay, device=mesh_device, mesh_mapper=rep
    )
    dense_w = up(dense.reshape(1, 1, T, E), ttnn.float32)
    idx_t = up(idx.reshape(1, 1, T, K).float(), ttnn.float32)
    w_t = up(w.reshape(1, 1, T, K), ttnn.float32)
    mask_t = up(mask, ttnn.float32)
    x = up(torch.randn(1, 1, T, H, generator=g), ttnn.bfloat16)
    tril = up(torch.tril(torch.ones(T, T, dtype=torch.bfloat16)).reshape(1, 1, T, T), ttnn.bfloat16)
    rank_cfg = _compute(mesh_device, ttnn.MathFidelity.HiFi4)
    max_buf = -(-(T * K + E_LOC * TILE) // TILE) * TILE

    def onehot_rank():
        onehot = ttnn.multiply(ttnn.typecast(ttnn.ne(dense_w, 0.0), ttnn.float32), mask_t)
        oh16 = ttnn.typecast(onehot, ttnn.bfloat16)
        rank = ttnn.matmul(tril, oh16, dtype=ttnn.float32, compute_kernel_config=rank_cfg)
        ttnn.deallocate(oh16)
        return onehot, rank

    def table(onehot, rank):
        counts_f = ttnn.slice(rank, [0, 0, T - 1, 0], [1, 1, T, E])
        aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(counts_f, TILE - 1), 1.0 / TILE)), float(TILE))
        incl = ttnn.cumsum(aligned, dim=-1)
        offsets_f = ttnn.subtract(incl, aligned)
        pos_te = ttnn.add(
            ttnn.multiply(ttnn.add(ttnn.subtract(rank, 1.0), offsets_f), onehot),
            ttnn.multiply(ttnn.rsub(onehot, 1.0), float(max_buf - 1)),
        )
        for t in (aligned, incl):
            ttnn.deallocate(t)
        idx_u32 = ttnn.typecast(idx_t, ttnn.uint32)
        pos_tk = ttnn.gather(pos_te, -1, idx_u32)
        loc_tk = ttnn.gather(onehot, -1, idx_u32)
        ttnn.deallocate(pos_te)
        ttnn.deallocate(idx_u32)
        counts = ttnn.reshape(ttnn.to_layout(ttnn.typecast(counts_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
        offsets = ttnn.reshape(ttnn.to_layout(ttnn.typecast(offsets_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
        ttnn.deallocate(counts_f)
        ttnn.deallocate(offsets_f)
        return pos_tk, loc_tk, counts, offsets

    def weights(loc_tk):
        w_loc = ttnn.multiply(w_t, loc_tk)
        w16 = ttnn.typecast(w_loc, ttnn.bfloat16)
        w_rm = ttnn.unsqueeze(ttnn.to_layout(w16, ttnn.ROW_MAJOR_LAYOUT), -1)
        ttnn.deallocate(w_loc)
        ttnn.deallocate(w16)
        return w_rm

    def x_rows():
        return ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [T, H])

    def g_onehot_rank():
        for t in onehot_rank():
            ttnn.deallocate(t)

    def g_all():
        onehot, rank = onehot_rank()
        outs = table(onehot, rank)
        ttnn.deallocate(onehot)
        ttnn.deallocate(rank)
        wr = weights(outs[1])
        for t in (*outs, wr):
            ttnn.deallocate(t)

    def g_x():
        ttnn.deallocate(x_rows())

    def timed(fn):
        for _ in range(2):
            fn()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            fn()
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / REPS * 1e3

    t_all, t_or, t_x = timed(g_all), timed(g_onehot_rank), timed(g_x)
    logger.info(
        f"[routing-ops] T={T} k={K} E={E} (72 local) max_buf={max_buf}, {REPS} back-to-back calls each: "
        f"routing table total {t_all:.3f} ms per sub-chunk (onehot + rank matmul {t_or:.3f} ms, the rest "
        f"{t_all - t_or:.3f} ms); x to rows {t_x:.3f} ms"
    )
