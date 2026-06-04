# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Expert-centric MoE Combine Module (PyTorch Reference Implementation)

This module implements the combine operation for Mixture-of-Experts (MoE) layers.
It reconstructs the original token ordering after expert processing.

Goals:
- Combine expert outputs back to original token positions
- Uses metadata from dispatch to correctly reassemble tokens
- Maintains expert-centric buffer organization during processing
"""

import torch

from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping


class TorchCombineModule(torch.nn.Module):
    """Expert-centric MoE combine module."""

    def __init__(
        self,
        dispatch_group_size: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
        num_dispatch_groups: int = 1,
    ):
        """
        Initialize combine module with configuration parameters.

        Args:
            dispatch_group_size: Number of chips in each dispatch group
            experts_per_chip: Number of experts per chip
            num_experts_per_tok: Number of experts each token is routed to
            seq_len_per_chip: Sequence length per chip
            num_dispatch_groups: Number of parallel dispatch groups
        """
        super().__init__()
        self.dispatch_group_size = dispatch_group_size
        self.experts_per_chip = experts_per_chip
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.num_dispatch_groups = num_dispatch_groups

    def forward(
        self,
        dispatched_buffer: torch.Tensor,
        metadata: torch.Tensor,
        expert_token_counts: torch.Tensor,
        expert_region_offsets: torch.Tensor,
    ):
        """
        Combine expert outputs back to original token positions.

        Args:
            dispatched_buffer: Dispatched tokens of shape (num_dispatch_groups, dispatch_group_size, max_dispatch_buffer_token_size, emb_dim)
            metadata: Metadata tensor containing token positions, same flat layout
            expert_token_counts: Counter tracking tokens per expert
            expert_region_offsets: Expert region offsets (shared across source devices in a
                dispatch group), shape (num_dispatch_groups, dispatch_group_size, num_routed_experts).
                Gives the expert region start position for each expert directly.

        Returns:
            y: Combined output tensor of shape (dispatch_group_size, seq_len, num_experts_per_tok, emb_dim)
        """
        # Infer emb_dim from dispatched tensor shape
        emb_dim = dispatched_buffer.shape[-1]

        y = torch.zeros(
            (self.dispatch_group_size, self.seq_len_per_chip, self.num_experts_per_tok, emb_dim),
            dtype=torch.bfloat16,
        )
        for group in range(self.num_dispatch_groups):
            for chip in range(self.dispatch_group_size):
                for expert in range(self.experts_per_chip):
                    global_expert_idx = ExpertMapping.get_global_expert_idx(
                        group=group,
                        chip=chip,
                        local_expert=expert,
                        experts_per_chip=self.experts_per_chip,
                        dispatch_group_size=self.dispatch_group_size,
                        num_dispatch_groups=self.num_dispatch_groups,
                        is_col_major=True,
                    )
                    start = int(expert_region_offsets[group, chip, global_expert_idx].item())
                    for i in range(expert_token_counts[group, 0, global_expert_idx]):
                        group_from_meta = int(metadata[group, chip, start + i, 0]) % self.num_dispatch_groups
                        if group != group_from_meta:
                            continue
                        src_chip = int(metadata[group, chip, start + i, 0]) // self.num_dispatch_groups
                        token = int(metadata[group, chip, start + i, 1])
                        topk_idx = int(metadata[group, chip, start + i, 2])
                        y[src_chip, token, topk_idx] = dispatched_buffer[group, chip, start + i]

        return y
