# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEGate (NemotronHTopkRouter) — TP=4 on QB 4-chip Blackhole.

Gate matmul runs on device (replicated). Logits ([1, 128]) are brought to
CPU for routing (sigmoid, bias correction, topk, normalisation); the result
is a dense [1, 128] routing-weight tensor with zeros for inactive experts,
uploaded back to device so ttnn.sparse_matmul can use it as the sparsity mask.

n_group=1, topk_group=1: group selection is trivial (all experts are in a
single group) so the group-mask step is a no-op.
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _host_rep, _upload

N_ROUTED_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 6
N_GROUP = 1
TOPK_GROUP = 1
ROUTED_SCALING_FACTOR = 2.5


def moe_gate_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [tokens, 2688] bf16 on device
    weight: torch.Tensor,  # [128, 2688] float32 CPU
    e_score_correction_bias: torch.Tensor,  # [128] float32 CPU
    n_routed_experts: int = N_ROUTED_EXPERTS,
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
) -> ttnn.Tensor:
    """Returns dense routing-weights [tokens, n_routed_experts] bfloat16 on device.

    Non-active expert positions are zero; active positions hold the normalised,
    scaled routing weights.  Shape is suitable as the sparsity tensor for
    ttnn.sparse_matmul.
    """
    tokens = hidden_states.shape[0]

    w_tt = _upload(weight.bfloat16(), mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    logits_tt = ttnn.linear(hidden_states, w_tt, transpose_b=True)
    logits = _host_rep(logits_tt, mesh_device, tokens).float()  # [tokens, 128]

    scores = torch.sigmoid(logits)
    bias = e_score_correction_bias.float()
    scores_for_choice = scores + bias.unsqueeze(0)

    # n_group=1: single group → straight topk over all 128 experts.
    topk_indices = torch.topk(scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False)[1]

    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor

    # Scatter to dense [tokens, 128] with zeros for inactive experts.
    routing_dense = torch.zeros(tokens, n_routed_experts, dtype=torch.float32)
    routing_dense.scatter_(1, topk_indices, topk_weights)

    # Replicate the routing-weight tensor on all TP devices — expert weights are
    # replicated (not sharded), so each device needs the full [tokens, 128] sparsity mask.
    return _upload(routing_dense.bfloat16(), mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
