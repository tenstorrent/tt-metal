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
    out = ttnn.embedding(ids_tt, w_tt)
    return ttnn.to_layout(out, ttnn.TILE_LAYOUT)


def embedding_forward_tt(
    mesh_device: MeshDevice,
    ids_tt: ttnn.Tensor,  # [B, S] uint32 already on device (pre-allocated for trace)
    weight: torch.Tensor,  # [vocab_size, hidden_size] bf16 CPU
) -> ttnn.Tensor:
    """Returns [B, S, hidden_size] bfloat16 replicated on all devices.

    Accepts pre-allocated device tensor for trace compatibility — caller must
    update it via ttnn.copy_host_to_device_tensor before each execute_trace.
    """
    w_tt = _upload(weight, mesh_device, None, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
    out = ttnn.embedding(ids_tt, w_tt)
    return ttnn.to_layout(out, ttnn.TILE_LAYOUT)
