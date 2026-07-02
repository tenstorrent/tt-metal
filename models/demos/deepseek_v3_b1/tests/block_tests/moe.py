# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MoE (Mixture of Experts) FFN — reference functions."""

import torch
import torch.nn.functional as F

from mlp import mlp_torch


# ---------------------------------------------------------------------------
# Gate — top-k routing
# ---------------------------------------------------------------------------

def gate_torch(
    hidden_states,
    *,
    w_router,
    top_k,
    score_func="softmax",
    n_group=1,
    topk_group=1,
    e_score_correction_bias=None,
    routed_scaling_factor=1.0,
    norm_topk_prob=True,
):
    """
    Top-k routing gate with configurable scoring function.
    Matches the official DeepSeek V3 Gate implementation.

    hidden_states: [b, s, h]
    w_router: [num_experts, h]
    top_k: int — number of experts per token
    score_func: "softmax" (Mixtral/OLMoE/Qwen3-MoE) or "sigmoid" (DeepSeek V3/Kimi K2.5)
    n_group: number of expert groups (applies to both score_func modes)
    topk_group: number of groups to keep
    e_score_correction_bias: optional [num_experts] bias added before group selection
    routed_scaling_factor: scale applied to final weights
    norm_topk_prob: normalize top-k weights before scaling

    Returns (topk_indices, topk_weights) each [b*s, top_k].
    """
    num_experts = w_router.shape[0]
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    router_logits = F.linear(x.float(), w_router.float())

    # 1) Score function
    if score_func == "softmax":
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float)
    else:
        scores = router_logits.sigmoid()
    original_scores = scores

    # 2) Optional bias (e.g. e_score_correction_bias in DeepSeek V3)
    scores_for_choice = scores
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)

    # 3) Group-based selection (applies to both softmax and sigmoid)
    if n_group > 1:
        scores_for_choice = scores_for_choice.view(-1, n_group, num_experts // n_group)
        if e_score_correction_bias is None:
            group_scores = scores_for_choice.amax(dim=-1)
        else:
            group_scores = scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = scores_for_choice.new_ones(x.size(0), n_group, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)
        scores_for_choice = scores_for_choice.masked_fill(group_mask.unsqueeze(-1), float("-inf"))
        scores_for_choice = scores_for_choice.flatten(1)

    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1)[1]
    topk_weights = original_scores.gather(1, topk_indices)

    # 4) Weight normalization and scaling
    if score_func == "sigmoid" or norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_indices, topk_weights.type_as(x)


gate_tt = gate_torch


# ---------------------------------------------------------------------------
# MoE — reference implementation
# ---------------------------------------------------------------------------

def moe_torch(
    hidden_states,
    *,
    w_router,
    w1,
    w2,
    w3,
    top_k,
    gate_kwargs=None,
    precomputed_routing=None,
    shared_mlp_kwargs=None,
    scale_input=False,
):
    """
    Top-k routing MoE FFN. Each expert is a SwiGLU MLP.

    hidden_states: [b, s, h]
    w_router: [num_experts, h]
    w1: [num_experts, moe_intermediate, h]  (gate_proj)
    w2: [num_experts, h, moe_intermediate]  (down_proj)
    w3: [num_experts, moe_intermediate, h]  (up_proj)
    top_k: int
    gate_kwargs: optional dict of extra kwargs forwarded to gate_torch
    precomputed_routing: optional (topk_idx, routing_weights) — bypasses gate_torch entirely
    shared_mlp_kwargs: optional dict(w_gate, w_up, w_down) for shared experts
    scale_input: if True, scale input before expert (Llama4); else scale output (default)
    """
    b, s, h = hidden_states.shape

    x = hidden_states.reshape(b * s, h)

    if precomputed_routing is not None:
        topk_idx, routing_weights = precomputed_routing
    else:
        topk_idx, routing_weights = gate_torch(
            hidden_states, w_router=w_router, top_k=top_k,
            **(gate_kwargs or {}),
        )

    out = torch.zeros_like(x)

    for e in range(w_router.shape[0]):
        # Find which (token, slot) pairs selected this expert
        mask = topk_idx == e  # [b*s, top_k]
        token_mask = mask.any(dim=-1)  # [b*s]
        if not token_mask.any():
            continue

        token_ids = token_mask.nonzero(as_tuple=True)[0]
        x_e = x[token_ids]  # [n_tokens, h]

        # Gather routing weights for this expert across top_k slots
        weights_e = (mask[token_ids] * routing_weights[token_ids]).sum(dim=-1, keepdim=True)

        if scale_input:
            x_e = x_e * weights_e

        o_e = mlp_torch(x_e, w_gate=w1[e], w_up=w3[e], w_down=w2[e])

        if scale_input:
            out[token_ids] += o_e
        else:
            out[token_ids] += o_e * weights_e

    if shared_mlp_kwargs is not None:
        out = out + mlp_torch(x, **shared_mlp_kwargs)

    return out.reshape(b, s, h)


moe_tt = moe_torch
