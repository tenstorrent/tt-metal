# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host <-> mesh layout of the residual stream, for the component hooks (never inside a forward).

One 5120-token chunk on the [8, 4] mesh: SP row r (axis 0) holds tokens [r * 640, (r + 1) * 640) (the server's
block-cyclic reshuffle ``(g // 640) % 8`` within one chunk), and TP column c (axis 1) holds a quarter of them,
rows [c * 160, (c + 1) * 160) of its row's 640. Each chip: [1, 1, 160, 4096] bf16.
"""

from __future__ import annotations

import torch

import ttnn


def residual_to_mesh(mesh, x: torch.Tensor):
    """x [S, H] -> each chip [1, 1, S / (sp * tp), H] bf16 TILE, token order (sp row, tp column, local row)."""
    sp, tp = tuple(mesh.shape)
    S, H = x.shape[-2], x.shape[-1]
    assert S % (sp * tp * ttnn.TILE_SIZE) == 0, f"{S} rows do not split into {sp}x{tp} tile-aligned blocks"
    x = x.reshape(sp, tp, S // (sp * tp), H).to(torch.bfloat16)
    return ttnn.from_torch(
        x,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)),
    )


def residual_to_host(mesh, t) -> torch.Tensor:
    """The inverse of residual_to_mesh: [S, H]."""
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)))
    return out.reshape(-1, out.shape[-1])
