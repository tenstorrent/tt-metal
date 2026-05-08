# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Routed ``Mistral4NaiveMoe`` forward with **expert matmuls on device** (``dense_mlp_bf16``).

Indexing / ``index_add_`` follow Hugging Face ``Mistral4NaiveMoe.forward``; only ``F.linear`` + SiLU path is
replaced with the same device stack as :func:`~models.tt_transformers.tt.mistral_small_4.dense_mlp.dense_mlp_bf16`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.tt_transformers.tt.mistral_small_4.dense_mlp import dense_mlp_bf16


def mistral4_naive_moe_routed_bf16(
    mesh_device,
    hidden_states_flat_nh: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: torch.nn.Module,
) -> torch.Tensor:
    """
    Same output shape and semantics as ``Mistral4NaiveMoe.forward`` (routed part only).

    Args:
        mesh_device: TT mesh device.
        hidden_states_flat_nh: ``[N, hidden]`` bf16/float32 (HF flattens ``B,S,H`` → ``N,H``).
        topk_indices: ``[N, top_k]`` int64 (expert ids per token slot).
        topk_weights: ``[N, top_k]`` bf16 (weights from ``route_tokens_to_experts``).
        experts: HF ``Mistral4NaiveMoe`` (``gate_up_proj``, ``down_proj``, ``act_fn``).

    Returns:
        ``[N, hidden]`` bf16 on host (same dtype as ``hidden_states_flat_nh`` after bf16 matmuls).
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4NaiveMoe

    if not isinstance(experts, Mistral4NaiveMoe):
        raise TypeError("mistral4_naive_moe_routed_bf16 expects Mistral4NaiveMoe")

    hidden_states_flat_nh = hidden_states_flat_nh.to(torch.bfloat16)
    topk_weights = topk_weights.to(torch.bfloat16)

    n_tokens = int(hidden_states_flat_nh.shape[0])
    hidden_dim = int(hidden_states_flat_nh.shape[1])
    inter = int(experts.intermediate_dim)
    num_experts = int(experts.num_experts)

    final = torch.zeros_like(hidden_states_flat_nh)

    expert_mask = F.one_hot(topk_indices.to(torch.long), num_classes=num_experts)
    expert_mask = expert_mask.permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    with torch.no_grad():
        for row in expert_hit:
            expert_idx = int(row[0].item())
            if expert_idx == num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states_flat_nh[token_idx]
            w_row = topk_weights[token_idx, top_k_pos]

            geu = experts.gate_up_proj[expert_idx]
            gate_w = geu[:inter, :].contiguous()
            up_w = geu[inter:, :].contiguous()
            down_w = experts.down_proj[expert_idx].contiguous()

            cur_bsh = current_state.unsqueeze(0)
            current_hidden = dense_mlp_bf16(mesh_device, cur_bsh, gate_w, up_w, down_w).squeeze(0)
            current_hidden = current_hidden * w_row.unsqueeze(-1).to(current_hidden.dtype)
            final.index_add_(0, token_idx, current_hidden.to(final.dtype))

    return final
