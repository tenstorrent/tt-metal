# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Data structures for Torch MoE intermediate values.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MoEIntermediates:
    """Data structure holding intermediate values from MoE forward pass for debugging."""

    # fmt: off
    gate_scores: Optional[torch.Tensor] = None            # (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    gate_indices: Optional[torch.Tensor] = None           # (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    gate_logits: Optional[torch.Tensor] = None            # (dispatch_group_size * seq_len_per_chip, num_routed_experts)
    expert_token_counts: Optional[torch.Tensor] = None    # (num_dispatch_groups, 1, num_routed_experts)
    expert_region_offsets: Optional[torch.Tensor] = None  # (num_dispatch_groups, dispatch_group_size, num_routed_experts)
    dispatched_buffer: Optional[torch.Tensor] = None      # (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
    metadata: Optional[torch.Tensor] = None               # (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, metadata_len)
    expert_outputs: Optional[torch.Tensor] = None         # Same shape as dispatched_buffer
    shared_output: Optional[torch.Tensor] = None          # (dispatch_group_size, seq_len_per_chip, emb_dim)
    combined_output: Optional[torch.Tensor] = None        # (dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim)
    routed_output: Optional[torch.Tensor] = None          # (dispatch_group_size, seq_len_per_chip, emb_dim)
    # fmt: on
