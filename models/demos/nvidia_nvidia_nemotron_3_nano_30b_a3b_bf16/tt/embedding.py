# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embedding block — TP=4 on QB 4-chip Blackhole."""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _host_rep, _upload


def embedding_forward(
    mesh_device: MeshDevice,
    input_ids: torch.Tensor,  # [B, S] int32/int64 CPU
    weight: torch.Tensor,  # [vocab_size, hidden_size] bf16 CPU
) -> torch.Tensor:
    B = input_ids.shape[0]
    # input_ids change every call — not cached
    ids_tt = ttnn.from_torch(
        input_ids.int(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    # embedding weight is static — cache on device after first upload
    w_tt = _upload(weight, mesh_device, None, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
    out_tt = ttnn.embedding(ids_tt, w_tt)
    return _host_rep(out_tt, mesh_device, B)
