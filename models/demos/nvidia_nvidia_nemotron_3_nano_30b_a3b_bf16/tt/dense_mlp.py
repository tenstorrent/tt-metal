# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DenseMLP block — bringup on QB (device 0).

relu2 (squared-ReLU) MLP with pre-RMSNorm and residual.
hidden_size=2688, intermediate_size=1856.
"""

import torch

import ttnn
from ttnn import MeshDevice

NORM_EPS = 1e-5

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def dense_mlp_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688]
    norm_weight: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    norm_eps: float = NORM_EPS,
) -> torch.Tensor:
    residual = hidden_states

    h_tt = ttnn.from_torch(
        hidden_states.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    w_tt = ttnn.from_torch(
        norm_weight.bfloat16().unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    normed_tt = ttnn.rms_norm(h_tt, epsilon=norm_eps, weight=w_tt)

    wu_tt = ttnn.from_torch(
        w_up.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=_R(mesh_device)
    )
    up_tt = ttnn.linear(normed_tt, wu_tt, transpose_b=True)
    act_tt = ttnn.pow(ttnn.relu(up_tt), 2)

    wd_tt = ttnn.from_torch(
        w_down.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=_R(mesh_device)
    )
    out_tt = ttnn.linear(act_tt, wd_tt, transpose_b=True)
    out = ttnn.to_torch(out_tt, mesh_composer=_C(mesh_device, dim=0)).bfloat16()

    return (residual + out).bfloat16()
