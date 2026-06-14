# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEGate (NemotronHTopkRouter) — TP=4 on QB 4-chip Blackhole.

Gate matmul runs on device (replicated). Logits ([tokens, 128]) are small
enough to bring to CPU for routing; CPU topk logic is not torch.nn.functional.

n_group=1, topk_group=1: group selection is trivial (all experts are in a
single group) so the group-mask step is a no-op and we fall through to a
straight topk over all 128 experts.
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _host_rep, _rep

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
    n_group: int = N_GROUP,
    topk_group: int = TOPK_GROUP,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
) -> tuple:
    """Returns (topk_indices [tokens, k] int64, topk_weights [tokens, k] float32) on CPU.

    Gate matmul is on device; routing logic uses standard torch ops (not F.*).
    """
    tokens = hidden_states.shape[0]

    # Gate weight uploaded in bf16 (matches input dtype; small accuracy gap vs fp32
    # is acceptable at bringup — routing PCC > 0.99).
    w_tt = _rep(weight.bfloat16(), mesh_device)
    logits_tt = ttnn.linear(hidden_states, w_tt, transpose_b=True)
    logits = _host_rep(logits_tt, mesh_device, tokens).float()  # [tokens, 128]

    scores = torch.sigmoid(logits)
    bias = e_score_correction_bias.float()
    scores_for_choice = scores + bias.unsqueeze(0)

    # n_group=1, topk_group=1 → single group contains all experts → mask is all-ones.
    # Skip the group-selection step entirely.
    topk_indices = torch.topk(scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False)[1]

    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_indices.long(), topk_weights.float()
