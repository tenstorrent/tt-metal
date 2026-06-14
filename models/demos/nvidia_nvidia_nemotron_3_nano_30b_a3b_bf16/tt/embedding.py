# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embedding block — TP=4 on QB 4-chip Blackhole."""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _upload


def embedding_forward(
    mesh_device: MeshDevice,
    input_ids: torch.Tensor,  # [B, S] int32/int64 CPU
    weight: torch.Tensor,  # [vocab_size, hidden_size] bf16 CPU
) -> ttnn.Tensor:
    """Returns [B, S, hidden_size] bfloat16 replicated on all devices."""
    ids_tt = ttnn.from_torch(
        input_ids.int(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w_tt = _upload(weight, mesh_device, None, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
    return ttnn.embedding(ids_tt, w_tt)
