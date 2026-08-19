# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host construction of KDA's fixed chunk-operation tensors."""

from __future__ import annotations

import torch

import ttnn

_CHUNK_SIZE = 32


def build_kda_const_tiles(
    device: ttnn.Device | ttnn.MeshDevice,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Build replicated eye, triangular, ones, and quadrant-mask tiles."""
    eye = torch.eye(_CHUNK_SIZE, dtype=torch.float32)
    tril = torch.tril(torch.ones(_CHUNK_SIZE, _CHUNK_SIZE, dtype=torch.float32))
    ones = torch.ones(_CHUNK_SIZE, _CHUNK_SIZE, dtype=torch.float32)

    # Packing must match make_quadrant_masks in chunk_gated_delta_rule.cpp.
    indices = torch.arange(_CHUNK_SIZE)
    rows = indices.unsqueeze(1)
    columns = indices.unsqueeze(0)
    lower_rows = rows < _CHUNK_SIZE / 2
    lower_columns = columns < _CHUNK_SIZE / 2
    masks = torch.cat(
        [
            (lower_rows & lower_columns).float(),
            (~lower_rows & ~lower_columns).float(),
            (~lower_rows & lower_columns).float(),
        ],
        dim=1,
    )

    def upload(tensor: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.reshape(1, 1, *tensor.shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    return upload(eye), upload(tril), upload(ones), upload(masks)
