# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEGate (NemotronHTopkRouter) — bringup on QB (device 0).

Gate is [128, 2688] float32. Router logit matmul on device 0,
topk selection on host (host logic is cheap for 128 experts).
"""

import torch

import ttnn
from ttnn import MeshDevice

N_ROUTED_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 6
N_GROUP = 1
TOPK_GROUP = 1
ROUTED_SCALING_FACTOR = 2.5

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


def moe_gate_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [tokens, 2688] bf16 CPU
    weight: torch.Tensor,  # [128, 2688] float32 CPU
    e_score_correction_bias: torch.Tensor,  # [128] float32 CPU
    n_routed_experts: int = N_ROUTED_EXPERTS,
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK,
    n_group: int = N_GROUP,
    topk_group: int = TOPK_GROUP,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
) -> tuple:
    """Returns (topk_indices [tokens, k] int64, topk_weights [tokens, k] float32)."""
    tokens = hidden_states.shape[0]

    h_tt = ttnn.from_torch(
        hidden_states.float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    w_tt = ttnn.from_torch(
        weight.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=_R(mesh_device)
    )
    logits_tt = ttnn.linear(h_tt, w_tt, transpose_b=True)
    logits = ttnn.to_torch(logits_tt, mesh_composer=_C(mesh_device, dim=0)).float()  # [tokens, 128]

    scores = torch.sigmoid(logits)
    bias = e_score_correction_bias.float()
    scores_for_choice = scores + bias.unsqueeze(0)

    scores_3d = scores_for_choice.view(tokens, n_group, n_routed_experts // n_group)
    group_scores = scores_3d.topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1).expand(-1, n_group, n_routed_experts // n_group).reshape(tokens, n_routed_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False)[1]

    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_indices.long(), topk_weights.float()
