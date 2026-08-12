# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
    # LatentMoE only (Kimi-K3): post-reduce, BEFORE the latent norm + up-projection, so still at the
    # latent width. Lets a routed_output PCC miss be localised to either side of that boundary --
    # without it, routed_output bundles the reduce, the norm and the up-projection together.
    # None when there is no latent space, in which case routed_output already is this tensor.
    latent_routed_output: Optional[ttnn.Tensor] = None  # (dispatch_group_size_per_device, seq_len_per_chip, routed_emb_dim_per_tp)
    # LatentMoE only (Kimi-K3): post down-projection, BEFORE dispatch. Mirrors the reference's field
    # of the same name. Without it the down-projection -- one of the two genuinely new device ops --
    # has no intermediate of its own, so an enter()-side defect surfaces only as a
    # latent_routed_output miss with nothing to attribute it to.
    latent_input: Optional[ttnn.Tensor] = None          # (dispatch_group_size_per_device, seq_len_per_chip, routed_emb_dim)
    # fmt: on
