# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1a: the 131072-row DeepSeek prefill (4096 tile-rows x 3584 per device, bf16, RM γ, 16-bit DEST,
interleaved) — the x re-read rung of the L1 ladder.

Host-memory-light twin of eval test_translated::test_ds_distributed_norm[random_weights-prefill-131072] (whose torch
reference over 2 x 131072 x 7168 needs ~95 GB of host RAM): bf16 inputs, reference and PCC per row block.
Opt-in (DRMS_LONG_PREFILL=1): ~1 min and ~20 GB of host RAM; the reread path itself is pinned on small shapes by
test_distributed_rms_norm_l1_fallbacks.py (reread-* rungs).
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

pytestmark = pytest.mark.skipif(
    os.environ.get("DRMS_LONG_PREFILL", "0") in ("", "0"), reason="opt-in: DRMS_LONG_PREFILL=1 (~20 GB host RAM)"
)

EPS = 1e-6
H, W_LOCAL = 131072, 3584
ROW_BLOCK = 8192


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def test_prefill_131072_reread(mesh_cache):
    mesh = mesh_cache.get()
    rows, cols = tuple(mesh.shape)
    cluster_axis = 1
    torch.manual_seed(0)
    xs = [[torch.randn(1, 1, H, W_LOCAL, dtype=torch.bfloat16) for _ in range(cols)] for _ in range(rows)]
    gs = [[torch.randn(W_LOCAL, dtype=torch.bfloat16) for _ in range(cols)] for _ in range(rows)]
    glob = torch.cat([torch.cat(xs[r], dim=-1) for r in range(rows)], dim=0)
    tx = ttnn.from_torch(
        glob,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, -1), mesh_shape=(rows, cols)),
    )
    del glob
    gglob = torch.cat(
        [torch.cat([gs[r][c].reshape(1, 1, W_LOCAL // 32, 32) for c in range(cols)], dim=-1) for r in range(rows)],
        dim=0,
    )
    tg = ttnn.from_torch(
        gglob,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, -1), mesh_shape=(rows, cols)),
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
    )
    out = distributed_rms_norm(tx, cluster_axis=cluster_axis, gamma=tg, epsilon=EPS, compute_kernel_config=cfg)
    devs = ttnn.get_device_tensors(out)
    got = [[ttnn.to_torch(devs[r * cols + c]) for c in range(cols)] for r in range(rows)]
    for r in range(rows):
        w = torch.cat([gs[r][c] for c in range(cols)]).float()
        for r0 in range(0, H, ROW_BLOCK):
            full = torch.cat([xs[r][c][..., r0 : r0 + ROW_BLOCK, :] for c in range(cols)], dim=-1).float()
            ref = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=w, eps=EPS)
            for c in range(cols):
                g = got[r][c][..., r0 : r0 + ROW_BLOCK, :].float()
                assert torch.isfinite(g).all(), f"device ({r},{c}) rows {r0}.. non-finite"
                pcc = _pcc(g, ref[..., c * W_LOCAL : (c + 1) * W_LOCAL])
                assert pcc >= 0.995, f"device ({r},{c}) rows {r0}..{r0 + ROW_BLOCK} PCC {pcc:.6f}"
