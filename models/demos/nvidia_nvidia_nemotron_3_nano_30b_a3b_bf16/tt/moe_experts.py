# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEExperts — correctness bringup: pure host PyTorch, no TTNN device calls.

128 routed NemotronHMLP experts, top-6 per token, intermediate=1856.
Device-side Expert Parallelism (EP) is deferred to the optimization phase.

mesh_device is accepted for API compatibility but is not used.
"""

import torch
import torch.nn.functional as F

from ttnn import MeshDevice

N_EXPERTS = 128
TOP_K = 6
HIDDEN_SIZE = 2688
MOE_INTERMEDIATE = 1856


def moe_experts_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [tokens, 2688] bf16 CPU (pre-normed)
    topk_indices: torch.Tensor,  # [tokens, 6] int64 CPU
    topk_weights: torch.Tensor,  # [tokens, 6] float32 CPU
    expert_up_weights: list,  # list of 128 [1856, 2688] bf16 CPU
    expert_down_weights: list,  # list of 128 [2688, 1856] bf16 CPU
) -> torch.Tensor:
    """Sequential expert loop on host (pure PyTorch).

    Activation: relu2  — relu(x)^2
    Returns [tokens, 2688] bfloat16 (CPU).
    """
    n_tokens = hidden_states.shape[0]
    final = torch.zeros(n_tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
    one_hot = F.one_hot(topk_indices, num_classes=N_EXPERTS).permute(2, 0, 1)

    for e in range(N_EXPERTS):
        token_idx, weight_idx = torch.where(one_hot[e])
        if token_idx.numel() == 0:
            continue

        x_e = hidden_states[token_idx].float()  # [ne, 2688]
        up = F.linear(x_e, expert_up_weights[e].float())  # [ne, 1856]
        act = F.relu(up) ** 2  # relu2
        out = F.linear(act, expert_down_weights[e].float())  # [ne, 2688]

        weights = topk_weights[token_idx, weight_idx].unsqueeze(-1).float()
        final.index_add_(0, token_idx, (out * weights).to(torch.bfloat16))

    return final
