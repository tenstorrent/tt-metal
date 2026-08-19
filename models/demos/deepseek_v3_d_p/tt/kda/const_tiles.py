# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host construction of KDA's fixed chunk-operation tensors."""

from __future__ import annotations

import torch

import ttnn

_CHUNK_SIZE = 32


def build_kda_const_tiles(
    device: ttnn.Device | ttnn.MeshDevice,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Build the replicated eye, lower-triangular, and ones tiles."""
    eye = torch.eye(_CHUNK_SIZE, dtype=torch.float32)
    tril = torch.tril(torch.ones(_CHUNK_SIZE, _CHUNK_SIZE, dtype=torch.float32))
    ones = torch.ones(_CHUNK_SIZE, _CHUNK_SIZE, dtype=torch.float32)

    def upload(tensor: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.reshape(1, 1, *tensor.shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    return upload(eye), upload(tril), upload(ones)
