# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 MoE router (gate).
# Extracted verbatim from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     - topkgating()        lines 232-336
#     - HunyuanTopKGate     lines 1092-1140
#
# Used as the golden reference for TT-Metal numeric validation of the
# MoE router (Softmax -> TopK -> [optional group-greedy] -> normalize).
#
# IMPORTANT — which path the model actually runs:
#   HunyuanMoE.forward() calls the gate with topk_impl='easy', i.e. the
#   `easy_topk` static method below, NOT the full `topkgating` capacity/
#   group-greedy path. So for "select the same N experts as PyTorch",
#   the TT port must match `easy_topk`:
#       gates = softmax(logits, dim=1)            # logits = wg(x); wg is fp32, bias-free
#       topk_weight, expert_index = topk(gates, moe_topk)
#       topk_weight /= clamp(topk_weight.sum(-1, keepdim=True), min=1e-8)
#   `topkgating` is retained here for completeness / the 'default' impl.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def topkgating(
    logits: Tensor,
    topk: int,
    group_limited_greedy: bool = False,
    n_group: int = None,
    topk_group: int = None,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    capacity_factor: float = 1.0,
    drop_tokens: bool = False,
):
    """DeepSeek-style top-k gating with optional group-limited greedy routing
    and capacity-based dispatch. Returns:
        [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts
    """
    logits = logits.float()
    gates = F.softmax(logits, dim=1)

    if group_limited_greedy:
        group_shape = list(gates.shape[:-1]) + [n_group, gates.shape[-1] // n_group]
        group_scores = gates.reshape(group_shape).max(dim=-1).values  # [n, n_group]
        group_idx = torch.topk(group_scores, topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = group_mask.unsqueeze(-1).expand(group_shape).reshape(list(gates.shape))  # [n, e]
        gates = gates.masked_fill(~score_mask.bool(), 0.0)

    num_experts = int(gates.shape[1])
    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [tokens_per_group, num_selected_experts].
    expert_gate, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [tokens_per_group, num_experts]
    expert_mask_aux = expert_mask.max(dim=-2)[0]
    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    l_aux = num_experts**2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)

    if drop_tokens:
        expert_capacity = int(max(topk, topk * gates.shape[0] // gates.shape[1]) * capacity_factor)
    else:
        expert_index_flat = expert_index.flatten()
        tokens_per_expert = torch.bincount(expert_index_flat, minlength=num_experts)
        expert_capacity = torch.max(tokens_per_expert).item()

    if norm_topk_prob and topk > 1:
        gates_s = torch.clamp(
            torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1),
            min=torch.finfo(gates.dtype).eps,
        )
        router_probs = gates / gates_s
    else:
        router_probs = gates * routed_scaling_factor

    # Make num_selected_experts the leading axis so top-1 choices have priority
    # over top-2, which have priority over top-3, etc.
    expert_index = torch.transpose(expert_index, 0, 1)
    expert_index = expert_index.reshape(-1)  # [num_selected_experts * tokens_per_group]

    # Create mask out of indices. Shape: [tokens * topk, num_experts].
    expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)
    exp_counts = torch.sum(expert_mask, dim=0).detach()

    # A token's priority within an expert buffer is the masked cumulative count.
    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1
    token_priority = token_priority.reshape((topk, -1, num_experts))
    token_priority = torch.transpose(token_priority, 0, 1)  # [tokens, topk, num_experts]
    token_priority = torch.max(token_priority, dim=1)[0]  # [tokens, num_experts]

    valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    # combine_weights: router probs scaled into per-expert capacity slots.
    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    exp_counts_capacity = torch.sum(dispatch_mask)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)

    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


class HunyuanTopKGate(nn.Module):
    """
    MoE router. `wg` is a bias-free Linear kept in fp32; hidden states are
    upcast to fp32 before the projection so routing is deterministic across
    activation dtypes.

    Forward signature:
        hidden_states: [bsz, seq_len, hidden_size]
        topk_impl: 'default' -> topkgating(...)  (returns the gating tuple)
                   'easy'    -> easy_topk(...)    (returns (topk_weight, expert_index))

    The production inference path uses topk_impl='easy'.

    Notes for the TT-Metal port:
    - Match `easy_topk` to get identical expert selection: softmax over the
      expert axis (dim=1 of the flattened [tokens, num_experts] logits), then
      topk, then renormalize the selected weights.
    - Keep the router projection / softmax in fp32 — argmax ties on bf16 logits
      can flip expert choices.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        moe_topk: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        group_limited_greedy: bool = False,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        capacity_factor: float = 1.0,
        moe_drop_tokens: bool = False,
        moe_random_routing_dropped_token: bool = False,
    ):
        super().__init__()
        self.moe_topk = moe_topk
        self.drop_tokens = moe_drop_tokens
        self.min_capacity = 8
        self.random_routing_dropped_token = moe_random_routing_dropped_token
        self.capacity_factor = capacity_factor
        self.wg = nn.Linear(hidden_size, num_experts, bias=False, dtype=torch.float32)

        # DeepSeek gating args
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.group_limited_greedy = group_limited_greedy

    def forward(self, hidden_states, topk_impl: str = "default"):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        if topk_impl == "default":
            gate_output = topkgating(
                logits,
                self.moe_topk,
                group_limited_greedy=self.group_limited_greedy,
                n_group=self.n_group,
                topk_group=self.topk_group,
                norm_topk_prob=self.norm_topk_prob,
                routed_scaling_factor=self.routed_scaling_factor,
                capacity_factor=self.capacity_factor,
                drop_tokens=self.drop_tokens,
            )
        elif topk_impl == "easy":
            gate_output = self.easy_topk(logits, self.moe_topk)
        else:
            raise ValueError(f"Unsupported topk_impl: {topk_impl}")

        return gate_output

    @staticmethod
    def easy_topk(logits, moe_topk):
        gates = F.softmax(logits, dim=1)
        topk_weight_1, expert_index = torch.topk(gates, moe_topk)
        weight_sums = topk_weight_1.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        topk_weight = topk_weight_1 / weight_sums

        return topk_weight, expert_index


# ---------------------------------------------------------------------------
# Quick numeric smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, H = 1, 256, 4096
    NUM_EXPERTS, TOPK = 64, 8

    x = torch.randn(B, S, H, dtype=torch.bfloat16)
    gate = HunyuanTopKGate(hidden_size=H, num_experts=NUM_EXPERTS, moe_topk=TOPK).eval()

    with torch.no_grad():
        topk_weight, expert_index = gate(x, topk_impl="easy")

    print(f"input        : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"expert_index : {tuple(expert_index.shape)}  dtype={expert_index.dtype}")
    print(f"topk_weight  : {tuple(topk_weight.shape)}  rowsum={topk_weight[0].sum():.6f}")
    print(f"token0 experts: {expert_index[0].tolist()}")
    print(f"token0 weights: {[round(w, 4) for w in topk_weight[0].tolist()]}")
