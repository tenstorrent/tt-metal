# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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
    ):
        """
        Combine expert outputs back to original token positions.

        Args:
            dispatched_buffer: Dispatched tokens of shape (dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
            metadata: Metadata tensor containing token positions
            expert_token_counts: Counter tracking tokens per expert

        Returns:
            y: Combined output tensor of shape (dispatch_group_size, seq_len, num_experts_per_tok, hidden_dim)
        """
        # Infer hidden_dim from dispatched tensor shape
        hidden_dim = dispatched_buffer.shape[-1]

        y = torch.zeros(
            (self.dispatch_group_size, self.seq_len_per_chip, self.num_experts_per_tok, hidden_dim),
            dtype=torch.bfloat16,
        )
        for group in range(self.num_dispatch_groups):
            for chip in range(self.dispatch_group_size):
                for expert in range(self.experts_per_chip):
                    for i in range(expert_token_counts[group, chip, expert]):
                        group_from_meta = int(metadata[group, chip, expert, i, 0]) % self.num_dispatch_groups
                        if group != group_from_meta:
                            continue
                        src_chip = int(metadata[group, chip, expert, i, 0]) // self.num_dispatch_groups
                        token = int(metadata[group, chip, expert, i, 1])
                        topk_idx = int(metadata[group, chip, expert, i, 2])
                        y[src_chip, token, topk_idx] = dispatched_buffer[group, chip, expert, i]

        return y
