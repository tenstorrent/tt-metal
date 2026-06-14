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

from .tp import _col, _host_rep, _rep_keyed, all_gather

NORM_EPS = 1e-5


def lm_head_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [B, S, 2688] bf16 on device
    norm_f_weight: torch.Tensor,  # [2688] bf16 CPU
    lm_head_weight: torch.Tensor,  # [131072, 2688] bf16 CPU
    norm_eps: float = NORM_EPS,
) -> torch.Tensor:
    """Returns [B, S, 131072] bfloat16 CPU (final output boundary)."""
    B = hidden_states.shape[0]

    w_tt = _rep_keyed(id(norm_f_weight), norm_f_weight.bfloat16().unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(hidden_states, epsilon=norm_eps, weight=w_tt)

    wl_tt = _col(lm_head_weight, mesh_device)  # [32768, 2688]/device
    logits_tt = ttnn.linear(normed_tt, wl_tt, transpose_b=True)  # [B, S, 32768]/device

    gathered_tt = all_gather(logits_tt, dim=2)  # [B, S, 131072] on all devices

    return _host_rep(gathered_tt, mesh_device, B)


def lm_head_forward_device(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [B, S, 2688] bf16 on device
    norm_f_weight: torch.Tensor,  # [2688] bf16 CPU
    lm_head_weight: torch.Tensor,  # [131072, 2688] bf16 CPU
    norm_eps: float = NORM_EPS,
) -> ttnn.Tensor:
    """Returns [B, S, 131072] bfloat16 on device (no D2H; for trace capture)."""
    w_tt = _rep_keyed(id(norm_f_weight), norm_f_weight.bfloat16().unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(hidden_states, epsilon=norm_eps, weight=w_tt)

    wl_tt = _col(lm_head_weight, mesh_device)  # [32768, 2688]/device
    logits_tt = ttnn.linear(normed_tt, wl_tt, transpose_b=True)  # [B, S, 32768]/device

    return all_gather(logits_tt, dim=2)  # [B, S, 131072] on all devices
