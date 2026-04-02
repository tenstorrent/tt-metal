# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Data structures for TTNN MoE intermediate values.
"""

from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class TtMoEIntermediates:
    """
    Data structure holding intermediate values from TtMoe forward pass for debugging.

    Fields set to None indicate that component is not yet enabled/calculated.
    """

    # fmt: off
    gate_scores: Optional[ttnn.Tensor] = None           # (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    gate_indices: Optional[ttnn.Tensor] = None          # (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    gate_logits: Optional[ttnn.Tensor] = None           # (dispatch_group_size * seq_len_per_chip, num_routed_experts)
    expert_token_counts: Optional[ttnn.Tensor] = None   # from gate routing setup
    dispatched_buffer: Optional[ttnn.Tensor] = None     # (1, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
    metadata: Optional[ttnn.Tensor] = None              # (1, dispatch_group_size, experts_per_chip, max_tokens, metadata_len)
    expert_outputs: Optional[ttnn.Tensor] = None        # Same shape as dispatched_buffer
    shared_output: Optional[ttnn.Tensor] = None         # (dispatch_group_size_per_device, seq_len_per_chip, emb_dim_per_tp)
    combined_output: Optional[ttnn.Tensor] = None       # (1, dispatch_group_size_per_device, seq_len_per_chip, num_experts_per_tok, emb_dim)
    routed_output: Optional[ttnn.Tensor] = None         # (dispatch_group_size_per_device, seq_len_per_chip, emb_dim_per_tp)
    # fmt: on
