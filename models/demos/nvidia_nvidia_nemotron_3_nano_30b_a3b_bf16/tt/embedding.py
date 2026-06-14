# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embedding block — bringup on QB (device 0)."""

import torch

import ttnn
from ttnn import MeshDevice

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def embedding_forward(
    mesh_device: MeshDevice,
    input_ids: torch.Tensor,  # [B, S] int32/int64 CPU
    weight: torch.Tensor,  # [vocab_size, hidden_size] bf16 CPU
) -> torch.Tensor:
    ids_tt = ttnn.from_torch(
        input_ids.int(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    w_tt = ttnn.from_torch(
        weight.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    out_tt = ttnn.embedding(ids_tt, w_tt)
    return ttnn.to_torch(out_tt, mesh_composer=_C(mesh_device, dim=0)).bfloat16()
