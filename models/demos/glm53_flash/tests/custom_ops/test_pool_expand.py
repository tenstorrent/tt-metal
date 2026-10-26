# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/pool_expand`` vs ``TtDSAIndexer._expand_pools``'s composite chain (tt-blaze G53F-0210).

For each T in ``PE_TS`` rows: per-chip random ranked pool ids < 262176 (the 1M geometry's pools) and tail rows mixing
token ids and sentinels (as ``_tail_ids`` builds them), sharded so every chip differs; the fused op must equal the
composite (typecast -> repeat_interleave -> multiply -> add offsets -> typecast -> concat tail + sentinels) bitwise on
every chip, and equal the host formula. Then both are timed (``PE_REPS`` back-to-back calls, one synchronize).

    pytest models/demos/glm53_flash/tests/custom_ops/test_pool_expand.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.pool_expand.op import PoolExpand
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
TS = [int(v) for v in os.environ.get("PE_TS", "2048,512,32").split(",")]
REPS = int(os.environ.get("PE_REPS", "20"))
K, P, POOLS = 512, 4, 262176
SENTINEL = 0xFFFFFFFF


def _composite(top_p, tail, T, offsets, sentinels):
    """``TtDSAIndexer._expand_pools`` op for op (k_eff = P * K)."""
    top_f = ttnn.typecast(top_p, ttnn.float32)
    top_rep = ttnn.repeat_interleave(top_f, P, dim=-1)
    top_tok = ttnn.add(ttnn.multiply(top_rep, float(P)), offsets)
    top = ttnn.typecast(top_tok, ttnn.uint32)
    return ttnn.concat([top, tail, sentinels], dim=-1)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_pool_expand(mesh_device):
    D = MESH[0] * MESH[1]
    g = torch.Generator().manual_seed(13)
    op = PoolExpand(mesh_device, pool=P, align=128)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    ok = True
    for T in TS:
        top_h = torch.randint(0, POOLS, (D, 1, T, K), generator=g, dtype=torch.int64)
        a = torch.randint(2048, 1048576, (D, 1, T, 1), generator=g, dtype=torch.int64)  # query positions
        base = (a + 1) // P * P
        tail_h = torch.stack([base + j for j in range(P - 1)], dim=-1).reshape(D, 1, T, P - 1)
        tail_h = torch.where(tail_h <= a, tail_h, torch.full_like(tail_h, SENTINEL))
        up = lambda t: ttnn.from_torch(
            t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=shard
        )
        top, tail = up(top_h), up(tail_h)
        W = op.width(K)
        offs = torch.arange(P, dtype=torch.float32).repeat(K).reshape(1, 1, 1, K * P).expand(1, 1, T, K * P)
        offsets = ttnn.from_torch(
            offs.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=rep
        )
        sentinels = ttnn.from_torch(
            torch.full((1, 1, T, W - K * P - (P - 1)), SENTINEL, dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        ref, got = _composite(top, tail, T, offsets, sentinels), op(top, tail)
        assert list(got.shape) == list(ref.shape), (got.shape, ref.shape)
        rows = lambda t: [
            ttnn.to_torch(c).reshape(T, W).to(torch.int64) & 0xFFFFFFFF for c in ttnn.get_device_tensors(t)
        ]
        r_c, g_c = rows(ref), rows(got)
        eq = sum(torch.equal(x, y) for x, y in zip(r_c, g_c))
        host = torch.cat(
            [
                (top_h[0, 0].unsqueeze(-1) * P + torch.arange(P)).reshape(T, K * P),
                tail_h[0, 0],
                torch.full((T, W - K * P - (P - 1)), SENTINEL, dtype=torch.int64),
            ],
            dim=-1,
        )
        host_eq = torch.equal(g_c[0], host)
        for t in (ref, got):
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

        ms_c = timed(lambda: _composite(top, tail, T, offsets, sentinels))
        ms_f = timed(lambda: op(top, tail))
        logger.info(
            f"[pool-expand] T={T}: fused == composite on {eq} / {len(r_c)} chips, == host formula (chip 0) {host_eq}; "
            f"composite {ms_c:.3f} ms, fused {ms_f:.3f} ms per call"
        )
        ok &= eq == len(r_c) and host_eq
        for t in (top, tail, offsets, sentinels):
            ttnn.deallocate(t)
    assert ok, "the fused index rows differ from the composite's"
