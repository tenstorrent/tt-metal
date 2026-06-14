# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm block — bringup on QB (device 0)."""

import torch

import ttnn
from ttnn import MeshDevice

NORM_EPS = 1e-5

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def layer_norm_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    eps: float = NORM_EPS,
) -> torch.Tensor:
    h_tt = ttnn.from_torch(
        hidden_states.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    w_tt = ttnn.from_torch(
        weight.bfloat16().unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    out_tt = ttnn.rms_norm(h_tt, epsilon=eps, weight=w_tt)
    return ttnn.to_torch(out_tt, mesh_composer=_C(mesh_device, dim=0)).bfloat16()
