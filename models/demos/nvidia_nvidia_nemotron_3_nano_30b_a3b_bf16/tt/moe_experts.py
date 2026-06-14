# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEExperts — TP=4 on QB 4-chip Blackhole.

128 routed NemotronHMLP experts, top-6 per token (top-k=6), intermediate=1856.
With top-6 of 128, on average only ~4.7% of experts process any token;
the loop skips all inactive experts immediately.

Each active expert's MLP (up_proj → relu2 → down_proj) runs on device via
ttnn.linear.  Routing logic (finding which tokens map to which expert) runs
on CPU using plain torch ops — no torch.nn.functional.

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

    # Bring activations to CPU once for token selection (not per-expert).
    flat_cpu = _host_rep(hidden_states, mesh_device, n_tokens).float()  # [tokens, 2688]

    final = torch.zeros(n_tokens, HIDDEN_SIZE, dtype=torch.float32)

    for e in range(N_EXPERTS):
        # Plain torch — not torch.nn.functional.
        token_idx, weight_idx = torch.where(topk_indices == e)
        if token_idx.numel() == 0:
            continue  # expert not active this step

        ne = token_idx.numel()
        x_e = flat_cpu[token_idx]  # [ne, 2688] CPU

        # Expert forward on device.
        x_e_tt = _rep(x_e.bfloat16(), mesh_device)
        wu_tt = _rep(expert_up_weights[e].bfloat16(), mesh_device)
        up_tt = ttnn.linear(x_e_tt, wu_tt, transpose_b=True)  # [ne, 1856]
        act_tt = ttnn.pow(ttnn.relu(up_tt), 2)  # relu2
        wd_tt = _rep(expert_down_weights[e].bfloat16(), mesh_device)
        out_tt = ttnn.linear(act_tt, wd_tt, transpose_b=True)  # [ne, 2688]

        out = _host_rep(out_tt, mesh_device, ne).float()  # [ne, 2688]
        weights = topk_weights[token_idx, weight_idx].unsqueeze(-1).float()
        final.index_add_(0, token_idx, out * weights)

    # Return result on device.
    return _rep(final.bfloat16(), mesh_device)
