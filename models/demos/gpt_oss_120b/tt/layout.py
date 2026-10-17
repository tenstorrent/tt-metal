# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host <-> mesh layout of the residual stream (plan.md "Mesh and axes"), for the component harness boundary only.

Mesh [8, 4]: axis 0 (rows) is SP, axis 1 (cols) is TP. A chunk's [S, H] hidden state is split by sequence over the
rows (chip row r holds rows r S/8 .. (r + 1) S/8, the engine's chunk order for S = 5120) and replicated over the
columns: every chip holds [1, 1, S/8, H] bf16 TILE in DRAM. Model forwards never call these.
"""

import torch

import ttnn


def rows_to_device(mesh, x: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """[S, H] host -> [1, 1, S/8, H] per chip, split over axis 0, replicated over axis 1."""
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]).to(torch.bfloat16),
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None)),
    )


def rows_to_host(mesh, t: ttnn.Tensor) -> torch.Tensor:
    """[1, 1, S/8, H] per chip (replicated over axis 1) -> [S, H] host fp32, read from column 0."""
    rows, cols = tuple(mesh.shape)
    dts = ttnn.get_device_tensors(t)
    parts = [ttnn.to_torch(dts[r * cols]).float() for r in range(rows)]
    out = torch.cat(parts, dim=-2)
    return out.reshape(out.shape[-2], out.shape[-1])
