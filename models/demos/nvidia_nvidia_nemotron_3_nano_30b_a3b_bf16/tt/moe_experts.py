# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEExperts — TP=4 on QB 4-chip Blackhole.

128 routed NemotronHMLP experts, top-6 per token (top-k=6), intermediate=1856.
With top-6 of 128, on average only ~4.7% of experts process any token;
the loop skips all inactive experts immediately.

Decode fast path (n_tokens == 1): all active experts process the same single
token, so hidden_states stays on device throughout — no D2H/H2D round trips.
Routing logic (which experts are active) still runs on CPU.

Prefill path (n_tokens > 1): original CPU-accumulation approach; activations
are brought to CPU once, each expert's slice is run on device, results
accumulated on CPU, and the final tensor is uploaded back to device.

Boundary: activations enter as ttnn.Tensor, result is returned as ttnn.Tensor.
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _host_rep, _rep

N_EXPERTS = 128
TOP_K = 6
HIDDEN_SIZE = 2688
MOE_INTERMEDIATE = 1856


def moe_experts_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [tokens, 2688] bf16 on device
    topk_indices: torch.Tensor,  # [tokens, 6] int64 CPU (from moe_gate)
    topk_weights: torch.Tensor,  # [tokens, 6] float32 CPU (from moe_gate)
    expert_up_weights: list,  # 128 × [1856, 2688] bf16 CPU
    expert_down_weights: list,  # 128 × [2688, 1856] bf16 CPU
) -> ttnn.Tensor:
    """Returns [tokens, 2688] bfloat16 on device (replicated).

    Activation: relu2 = relu(x)^2
    """
    n_tokens = hidden_states.shape[0]

    if n_tokens == 1:
        return _decode_experts_forward(
            mesh_device,
            hidden_states,
            topk_indices,
            topk_weights,
            expert_up_weights,
            expert_down_weights,
        )

    # --- Prefill: multi-token path (original logic) ---
    flat_cpu = _host_rep(hidden_states, mesh_device, n_tokens).float()  # [tokens, 2688]

    final = torch.zeros(n_tokens, HIDDEN_SIZE, dtype=torch.float32)

    for e in range(N_EXPERTS):
        token_idx, weight_idx = torch.where(topk_indices == e)
        if token_idx.numel() == 0:
            continue

        ne = token_idx.numel()
        x_e = flat_cpu[token_idx]  # [ne, 2688] CPU

        x_e_tt = _rep(x_e.bfloat16(), mesh_device)
        wu_tt = _rep(expert_up_weights[e].bfloat16(), mesh_device)
        up_tt = ttnn.linear(x_e_tt, wu_tt, transpose_b=True)  # [ne, 1856]
        act_tt = ttnn.pow(ttnn.relu(up_tt), 2)
        wd_tt = _rep(expert_down_weights[e].bfloat16(), mesh_device)
        out_tt = ttnn.linear(act_tt, wd_tt, transpose_b=True)  # [ne, 2688]

        out = _host_rep(out_tt, mesh_device, ne).float()  # [ne, 2688]
        weights = topk_weights[token_idx, weight_idx].unsqueeze(-1).float()
        final.index_add_(0, token_idx, out * weights)

    return _rep(final.bfloat16(), mesh_device)


def _decode_experts_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [1, 2688] bf16 on device
    topk_indices: torch.Tensor,  # [1, 6] int64 CPU
    topk_weights: torch.Tensor,  # [1, 6] float32 CPU
    expert_up_weights: list,
    expert_down_weights: list,
) -> ttnn.Tensor:
    """Decode fast path: single token — all experts process the same input.

    No D2H/H2D for activations.  Routing (which experts, what weights) is
    determined on CPU; only weight uploads and device compute are issued.
    """
    result_tt = None

    for e in range(N_EXPERTS):
        token_idx, weight_idx = torch.where(topk_indices == e)
        if token_idx.numel() == 0:
            continue

        # S=1: token_idx is always [0] — input is always the full hidden_states.
        w = topk_weights[0, weight_idx[0]].item()

        wu_tt = _rep(expert_up_weights[e].bfloat16(), mesh_device)
        up_tt = ttnn.linear(hidden_states, wu_tt, transpose_b=True)  # [1, 1856]
        act_tt = ttnn.pow(ttnn.relu(up_tt), 2)
        wd_tt = _rep(expert_down_weights[e].bfloat16(), mesh_device)
        out_tt = ttnn.linear(act_tt, wd_tt, transpose_b=True)  # [1, 2688]

        # Scale by routing weight — multiply by Python float (no extra tensor).
        out_tt = ttnn.multiply(out_tt, w)

        if result_tt is None:
            result_tt = out_tt
        else:
            result_tt = ttnn.add(result_tt, out_tt)

    return result_tt
