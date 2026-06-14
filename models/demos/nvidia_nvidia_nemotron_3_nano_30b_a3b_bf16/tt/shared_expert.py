# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SharedExpert — bringup on QB (device 0).

relu2 MLP, intermediate=3712 (moe_shared_expert_intermediate_size).
Input is pre-normed (no pre-norm or residual here).
"""

import torch

import ttnn
from ttnn import MeshDevice

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def shared_expert_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688] bf16 CPU (pre-normed)
    w_up: torch.Tensor,  # [3712, 2688] bf16 CPU
    w_down: torch.Tensor,  # [2688, 3712] bf16 CPU
) -> torch.Tensor:
    h_tt = ttnn.from_torch(
        hidden_states.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    wu_tt = ttnn.from_torch(
        w_up.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=_R(mesh_device)
    )
    up_tt = ttnn.linear(h_tt, wu_tt, transpose_b=True)
    act_tt = ttnn.pow(ttnn.relu(up_tt), 2)

    wd_tt = ttnn.from_torch(
        w_down.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=_R(mesh_device)
    )
    out_tt = ttnn.linear(act_tt, wd_tt, transpose_b=True)
    return ttnn.to_torch(out_tt, mesh_composer=_C(mesh_device, dim=0)).bfloat16()
