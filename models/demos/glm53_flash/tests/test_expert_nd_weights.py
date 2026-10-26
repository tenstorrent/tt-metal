# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regions-MoE expert kernel with DRAM ND-sharded expert weights vs the interleaved ones (tt-blaze G53F-0204).

``unified_routed_expert_moe`` coalesces a core's per-K-row weight slice into ONE NoC request when the weight is DRAM
ND-sharded one tile-row tall and ``per_core_N`` wide (``deepseek_v3_d_p/tt/moe/tt_routed_expert.py``
``routed_expert_weight_memory_config``; the d_p model's Blackhole default), one request per tile when it is
interleaved -- the GLM regions path's per-expert ``ttnn.slice`` copies are interleaved. At the 8x4 per-chip shapes
(72 local experts, gate / up [4096, 256], down [256, 4096] bf8, one 8192-token sub-chunk x top-8 routed uniformly,
the column-0 chip's experts) this times the op with the interleaved weights, with ND-sharded down only (gate / up are
8 tiles wide: per_core_N = 1 on the 11-wide grid, no run to coalesce) and with all three ND-sharded, and compares
every expert's region rows bitwise with the interleaved output on every chip.

    pytest models/demos/glm53_flash/tests/test_expert_nd_weights.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import routed_expert_weight_memory_config
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.moe import _compute

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("EXND_T", "8192"))
K, E, E_LOC, H, I, TILE = 8, 288, 72, 4096, 256, 32
REPS = int(os.environ.get("EXND_REPS", "10"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_expert_nd_weights(mesh_device):
    g = torch.Generator().manual_seed(3)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    counts = torch.zeros(E, dtype=torch.int64)
    for e in idx[idx < E_LOC].tolist():
        counts[e] += 1
    aligned = (counts + TILE - 1) // TILE * TILE
    offsets = torch.cumsum(aligned, 0) - aligned
    max_buf = -(-(T * K + E_LOC * TILE) // TILE) * TILE
    run = int(offsets[E - 1] + aligned[E - 1])
    up_rm = lambda t, dt: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    x = up_rm(torch.randn(max_buf, H, generator=g).to(torch.bfloat16), ttnn.bfloat16)
    counts_t = up_rm(counts.to(torch.int32).reshape(1, E), ttnn.uint32)
    offsets_t = up_rm(offsets.to(torch.int32).reshape(1, E), ttnn.uint32)
    gidx = up_rm(torch.arange(E_LOC, dtype=torch.int32).reshape(1, E_LOC), ttnn.uint32)

    def weight(shape, scale):
        return ttnn.from_torch(
            torch.randn(*shape, generator=g) * scale,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )

    gate = [weight((H, I), 0.02) for _ in range(E_LOC)]
    up = [weight((H, I), 0.02) for _ in range(E_LOC)]
    down = [weight((I, H), 0.05) for _ in range(E_LOC)]
    nd = lambda ws, n: [ttnn.to_memory_config(w, routed_expert_weight_memory_config(mesh_device, n, True)) for w in ws]
    down_nd = nd(down, H)
    gate_nd, up_nd = nd(gate, I), nd(up, I)
    compute = _compute(mesh_device)

    def call(gs, us, ds):
        return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x,
            offsets_t,
            counts_t,
            gidx,
            gs,
            us,
            ds,
            T,
            compute_kernel_config=compute,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        )

    rows = torch.cat([torch.arange(int(offsets[e]), int(offsets[e] + counts[e])) for e in range(E_LOC)])
    variants = {"interleaved": (gate, up, down), "nd_down": (gate, up, down_nd), "nd_all": (gate_nd, up_nd, down_nd)}
    ref = None
    for name, ws in variants.items():
        y = call(*ws)
        chips = [ttnn.to_torch(t).reshape(max_buf, H)[rows] for t in ttnn.get_device_tensors(y)]
        ttnn.deallocate(y)
        if ref is None:
            ref = chips
        eq = sum(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(chips, ref))
        for _ in range(2):
            ttnn.deallocate(call(*ws))
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(call(*ws))
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / REPS * 1e3
        logger.info(
            f"[expert-nd] {name}: {ms:.3f} ms per call (T {T}, {int(counts[:E_LOC].sum())} local picks, run {run} of "
            f"{max_buf} rows); region rows == interleaved on {eq} / {len(chips)} chips"
        )
