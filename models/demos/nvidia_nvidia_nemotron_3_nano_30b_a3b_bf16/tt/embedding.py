# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embedding block — TP=4 on QB 4-chip Blackhole."""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _host_rep


def embedding_forward(
    mesh_device: MeshDevice,
    input_ids: torch.Tensor,  # [B, S] int32/int64 CPU
    weight: torch.Tensor,  # [vocab_size, hidden_size] bf16 CPU
) -> torch.Tensor:
    B = input_ids.shape[0]
    ids_tt = ttnn.from_torch(
        input_ids.int(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w_tt = ttnn.from_torch(
        weight.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = ttnn.embedding(ids_tt, w_tt)
    return _host_rep(out_tt, mesh_device, B)
