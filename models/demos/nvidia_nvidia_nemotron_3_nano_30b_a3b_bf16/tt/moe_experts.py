# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEExperts — bringup on QB (device 0).

128 routed NemotronHMLP experts, top-6 per token, intermediate=1856.
Uses device 0 for all expert computations (EP parallelism added later).
"""

import torch
import torch.nn.functional as F

import ttnn
from ttnn import MeshDevice

N_EXPERTS = 128
TOP_K = 6
HIDDEN_SIZE = 2688
MOE_INTERMEDIATE = 1856

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def moe_experts_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [tokens, 2688] bf16 CPU (pre-normed)
    topk_indices: torch.Tensor,  # [tokens, 6] int64 CPU
    topk_weights: torch.Tensor,  # [tokens, 6] float32 CPU
    expert_up_weights: list,  # list of 128 [1856, 2688] bf16 CPU
    expert_down_weights: list,  # list of 128 [2688, 1856] bf16 CPU
) -> torch.Tensor:
    """Bringup: sequential expert loop on device 0.

    Returns [tokens, 2688] bfloat16 (CPU).
    """
    n_tokens = hidden_states.shape[0]
    final = torch.zeros(n_tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
    one_hot = F.one_hot(topk_indices, num_classes=N_EXPERTS).permute(2, 0, 1)

    for e in range(N_EXPERTS):
        token_idx, weight_idx = torch.where(one_hot[e])
        if token_idx.numel() == 0:
            continue

        x_e = hidden_states[token_idx]  # [ne, 2688]
        h_tt = ttnn.from_torch(
            x_e.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=_R(mesh_device),
        )
        wu_tt = ttnn.from_torch(
            expert_up_weights[e].bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=_R(mesh_device),
        )
        up_tt = ttnn.linear(h_tt, wu_tt, transpose_b=True)
        act_tt = ttnn.pow(ttnn.relu(up_tt), 2)
        wd_tt = ttnn.from_torch(
            expert_down_weights[e].bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=_R(mesh_device),
        )
        out_tt = ttnn.linear(act_tt, wd_tt, transpose_b=True)
        out = ttnn.to_torch(out_tt, mesh_composer=_C(mesh_device, dim=0)).bfloat16()

        weights = topk_weights[token_idx, weight_idx].unsqueeze(-1).float()
        final.index_add_(0, token_idx, (out * weights).to(torch.bfloat16))

    return final
