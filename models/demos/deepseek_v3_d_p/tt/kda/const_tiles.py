# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host construction of KDA's fixed grouped-recurrence identity tensor."""

from __future__ import annotations

import torch

import ttnn

_CHUNK_SIZE = 32


def build_kda_identity_tile(device: ttnn.Device | ttnn.MeshDevice) -> ttnn.Tensor:
    """Build the replicated identity tile used to summarize recurrence groups."""
    eye = torch.eye(_CHUNK_SIZE, dtype=torch.float32)
    return ttnn.from_torch(
        eye.reshape(1, 1, *eye.shape),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
