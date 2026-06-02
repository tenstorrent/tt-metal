# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shaw `relative_key` position-bias table for the Conformer self-attention.

The reference computes (modeling_seamless_m4t_v2.py:428-440):
    distance = clamp(r - l, -left, right)                    # (q_len, k_len)
    positional_embedding = distance_embedding[distance + left]  # (q_len, k_len, head_dim)
    rel = einsum("bhld,lrd->bhlr", query, positional_embedding) / sqrt(head_dim)

`positional_embedding` depends only on sequence length (not on data), so it is
precomputed once per seq_len on host and uploaded. The `einsum` itself is then a
device matmul (see conformer_attention.py), identical to the SpeechT5 encoder's
relative-position-bias path.
"""

from __future__ import annotations

import torch


def build_position_bias(seq_len: int, distance_embedding_weight: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Return positional_embedding of shape (seq_len, seq_len, head_dim).

    distance_embedding_weight: (left + right + 1, head_dim)
    """
    device = distance_embedding_weight.device
    pos_l = torch.arange(seq_len, device=device).view(-1, 1)
    pos_r = torch.arange(seq_len, device=device).view(1, -1)
    distance = torch.clamp(pos_r - pos_l, -left, right)
    idx = distance + left  # (seq_len, seq_len), in [0, left+right]
    return distance_embedding_weight[idx]  # (seq_len, seq_len, head_dim)
