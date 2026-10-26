# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/gather_combine_reduce`` vs the regions MoE's ``ttnn.embedding`` + ``post_combine_reduce`` (tt-blaze
G53F-0190), on the 8x4 mesh at the regions path's shapes: a [R, 4096] bf16 expert-output table whose last row is the
dummy slot, T x k picks of which ~1/4 are local (the rest at the dummy slot with weight 0), positive weights for the
local picks. Checks bitwise equality on every chip and times both (5 warm calls each).

    GCR_T=8192 pytest models/demos/glm53_flash/tests/custom_ops/test_gather_combine_reduce.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.gather_combine_reduce.op import GatherCombineReduce
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("GCR_T", "8192"))
K = 8
H = int(os.environ.get("GCR_H", "4096"))
LOCAL = float(os.environ.get("GCR_LOCAL", "0.25"))
REPS = int(os.environ.get("GCR_REPS", "5"))


def _inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    local = torch.rand(T, K, generator=g) < LOCAL
    n_local = int(local.sum())
    R = n_local + 64 * 32 + 1  # used rows + padding regions + the dummy slot (the regions buffer's shape)
    pos = torch.full((T, K), R - 1, dtype=torch.int64)
    pos[local] = torch.randperm(R - 1, generator=g)[:n_local]
    w = torch.rand(T, K, generator=g) * 0.5 + 0.01
    w[~local] = 0.0
    table = torch.randn(R, H, generator=g).to(torch.bfloat16)
    return table, pos, w.to(torch.bfloat16)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_gather_combine_reduce(mesh_device):
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    table_h, pos_h, w_h = _inputs(0)
    up = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh_device, mesh_mapper=rep)
    table = up(table_h, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    pos = up(pos_h.reshape(1, T * K).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    w_rm = up(w_h.reshape(1, 1, T, K, 1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    op = GatherCombineReduce(mesh_device)

    def reference():
        comb = ttnn.embedding(pos, table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, T*k, H]
        comb5 = ttnn.reshape(comb, [1, 1, T, K, H])
        y = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb5, w_rm, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(comb)
        return y

    def fused():
        return op(table, pos, w_rm)

    ref, got = reference(), fused()
    rd, gd = ttnn.get_device_tensors(ref), ttnn.get_device_tensors(got)
    eq = [torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)) for a, b in zip(rd, gd)]
    r0, g0 = ttnn.to_torch(rd[0]).float(), ttnn.to_torch(gd[0]).float()
    torch_ref = (table_h.float()[pos_h] * w_h.float().unsqueeze(-1)).sum(1)  # loose check of the reference itself
    logger.info(
        f"[gcr] T={T} k={K} H={H} R={int(table_h.shape[0])}: fused == reference bitwise on {sum(eq)}/{len(eq)} chips; "
        f"max|d| {(g0 - r0).abs().max().item():.3e}; reference vs torch max|d| "
        f"{(r0.reshape(T, H) - torch_ref).abs().max().item():.3e}"
    )
    ttnn.deallocate(ref)
    ttnn.deallocate(got)

    def timed(fn):
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / REPS * 1e3

    t_ref, t_fused = timed(reference), timed(fused)
    logger.info(f"[gcr] T={T}: embedding + post_combine_reduce {t_ref:.3f} ms, fused {t_fused:.3f} ms per call")
    assert all(eq), eq
