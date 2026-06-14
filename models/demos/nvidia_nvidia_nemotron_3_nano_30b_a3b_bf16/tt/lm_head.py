# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LMHead — final RMSNorm + vocab projection, TP=4 on QB 4-chip Blackhole.

Config: hidden_size=2688, vocab_size=131072, tie_word_embeddings=False.

TP strategy:
  norm_f: replicated
  lm_head [131072, 2688]: column-parallel → [32768, 2688]/device
  all_gather along dim=2 → [B, S, 131072] on all devices
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _col, _host_rep, _rep, all_gather

NORM_EPS = 1e-5


def lm_head_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688] bf16 CPU
    norm_f_weight: torch.Tensor,  # [2688] bf16 CPU
    lm_head_weight: torch.Tensor,  # [131072, 2688] bf16 CPU
    norm_eps: float = NORM_EPS,
) -> torch.Tensor:
    """Returns [B, S, 131072] bfloat16 (CPU)."""
    B = hidden_states.shape[0]

    # 1. Final RMSNorm: replicated → normed_tt [B, S, 2688] on all devices
    h_tt = _rep(hidden_states, mesh_device)
    w_tt = _rep(norm_f_weight.unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(h_tt, epsilon=norm_eps, weight=w_tt)

    # 2. Vocab projection: column-parallel → [B, S, 32768]/device
    wl_tt = _col(lm_head_weight, mesh_device)  # [32768, 2688]/device
    logits_tt = ttnn.linear(normed_tt, wl_tt, transpose_b=True)  # [B, S, 32768]/device

    # 3. All-gather along dim=2 → [B, S, 131072] on all devices
    gathered_tt = all_gather(logits_tt, dim=2)

    # 4. Bring to host; take first B rows from the replicated result
    return _host_rep(gathered_tt, mesh_device, B)
