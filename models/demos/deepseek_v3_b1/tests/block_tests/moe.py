"""MoE (Mixture of Experts) FFN — reference functions."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MoE — reference implementation
# ---------------------------------------------------------------------------

def moe_torch(
    hidden_states,
    *,
    w_router,
    w1,
    w2,
    top_k,
    w3=None,
    activation="silu",
    normalize_topk=True,
    precomputed_routing=None,
    scale_input=False,
):
    """
    Top-k routing MoE FFN.

    hidden_states: [b, s, h]
    w_router: [num_experts, h]
    w1: [num_experts, moe_intermediate, h]  (gate_proj for SwiGLU)
    w2: [num_experts, h, moe_intermediate]  (down_proj for SwiGLU)
    w3: [num_experts, moe_intermediate, h]  (up_proj for SwiGLU, optional)
    top_k: int
    precomputed_routing: optional (topk_idx, routing_weights) to bypass internal routing
    scale_input: if True, scale input before expert (Llama4); else scale output (default)
    """
    act_fn = F.silu if activation == "silu" else F.relu
    b, s, h = hidden_states.shape
    num_experts = w_router.shape[0]

    x = hidden_states.reshape(b * s, h)

    if precomputed_routing is not None:
        topk_idx, routing_weights = precomputed_routing
    else:
        logits = x @ w_router.T  # [b*s, num_experts]
        topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)  # [b*s, top_k]
        if normalize_topk:
            routing_weights = F.softmax(topk_vals, dim=-1)
        else:
            routing_weights = F.softmax(logits, dim=-1).gather(-1, topk_idx)

    out = torch.zeros_like(x)

    for e in range(num_experts):
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

        # Expert MLP: SwiGLU when w3 provided, else standard 2-layer
        gate_e = act_fn(F.linear(x_e, w1[e]))  # [n_tokens, intermediate]
        if w3 is not None:
            up_e = F.linear(x_e, w3[e])
            o_e = F.linear(gate_e * up_e, w2[e])
        else:
            o_e = F.linear(gate_e, w2[e])  # [n_tokens, h]

        if scale_input:
            out[token_ids] += o_e
        else:
            out[token_ids] += o_e * weights_e

    return out.reshape(b, s, h)


moe_tt = moe_torch


# ---------------------------------------------------------------------------
# Shared experts — reference implementation
# ---------------------------------------------------------------------------

def moe_shared_experts_torch(
    hidden_states,
    *,
    w_gate,
    w_up,
    w_down,
    activation="silu",
):
    """
    Shared-expert SwiGLU MLP applied to all tokens.

    hidden_states: [b, s, h]
    w_gate: [intermediate, h]
    w_up:   [intermediate, h]
    w_down: [h, intermediate]
    """
    act_fn = F.silu if activation == "silu" else F.relu
    gate = act_fn(F.linear(hidden_states, w_gate))
    up = F.linear(hidden_states, w_up)
    return F.linear(gate * up, w_down)


moe_shared_experts_tt = moe_shared_experts_torch


# ---------------------------------------------------------------------------
# Routing — sigmoid top-k (DeepSeek V3 / Kimi K2.5)
# ---------------------------------------------------------------------------

def sigmoid_topk_routing(
    hidden_states,
    *,
    w_router,
    top_k,
    n_group,
    topk_group,
    e_score_correction_bias=None,
    routed_scaling_factor=1.0,
    norm_topk_prob=False,
):
    """
    Sigmoid routing with group-based top-k selection (DeepSeek V3 pattern).

    hidden_states: [b, s, h]
    w_router: [num_experts, h]
    top_k: int — number of experts per token
    n_group: int — number of expert groups
    topk_group: int — number of groups to keep
    e_score_correction_bias: optional [num_experts] bias added before group selection
    routed_scaling_factor: float — scale applied to final weights
    norm_topk_prob: bool — normalize top-k weights before scaling

    Returns (topk_indices, topk_weights) each [b*s, top_k].
    """
    num_experts = w_router.shape[0]
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    router_logits = F.linear(x, w_router.float())
    scores = router_logits.sigmoid()

    # Group-based top-k selection
    scores_for_choice = scores.clone()
    if e_score_correction_bias is not None:
        scores_for_choice = scores_for_choice + e_score_correction_bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(-1, n_group, num_experts // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, num_experts // n_group)
        .reshape(-1, num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]

    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_indices, topk_weights
