# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Expert-centric MoE Dispatch Module (PyTorch Reference Implementation)

This module implements token dispatching for Mixture-of-Experts (MoE) layers.

Goals:
- Expert-centric buffer organization: [chips, experts_per_chip, tokens, hidden]
- Dense expert matmuls with no wasted compute (each expert only processes its routed tokens)
- No wasted memory (compact buffers, no sparse token arrays)
- Capacity factor (CF) handles load imbalance: allocates CF × expected_load per expert
- Full metadata tracking for round-trip verification: dispatch → experts → combine
"""

import torch
from loguru import logger


class TorchDispatchModule(torch.nn.Module):
    """Expert-centric MoE dispatch module."""

    def __init__(
        self,
        dispatch_group_size: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        seq_len_per_chip: int,
        hidden_dim: int = 7 * 1024,
        num_dispatch_groups: int = 1,
        expert_dispatch_table: torch.Tensor = None,
    ):
        """
        Initialize dispatch module with configuration parameters.

        Args:
            dispatch_group_size: Number of chips in each dispatch group
            experts_per_chip: Number of experts per chip
            num_routed_experts: Total number of routed experts across all chips
            metadata_len: Length of metadata per token (stores: chip, token, topk_idx, routed_expert, weight)
            max_dispatched_tokens_per_expert: Maximum number of tokens that can be dispatched to each expert
            expert_dispatch_table: Optional dispatch table of shape (num_dispatch_groups, num_routed_experts)
                Maps expert ID to logical chip ID in dispatch axis, -1 if not present
        """
        super().__init__()
        self.dispatch_group_size = dispatch_group_size
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.metadata_len = metadata_len
        self.max_dispatched_tokens_per_expert = max_dispatched_tokens_per_expert
        self.seq_len_per_chip = seq_len_per_chip
        self.num_dispatch_groups = num_dispatch_groups
        self.expert_dispatch_table = expert_dispatch_table

        # Oversized buffer (max_dispatched_tokens_per_expert) to simplify dispatch logic
        self.dispatched_shape = (
            num_dispatch_groups,
            dispatch_group_size,
            self.experts_per_chip,
            self.max_dispatched_tokens_per_expert,
            hidden_dim,
        )
        self.dispatched_metadata_shape = (
            num_dispatch_groups,
            dispatch_group_size,
            self.experts_per_chip,
            self.max_dispatched_tokens_per_expert,
            self.metadata_len,
        )

        self.dispatched_buffer = torch.zeros(self.dispatched_shape, dtype=torch.float32)
        self.dispatched_metadata = torch.ones(self.dispatched_metadata_shape, dtype=torch.int32) * -1

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_offsets: torch.Tensor,
    ):
        """
        Route tokens from their original positions to expert-specific buffers distributed across chips.

        Simulates MoE dispatch: each token is routed to multiple experts based on router indices.
        Tokens are gathered into per-expert buffers with metadata tracking their origin for later recombination.

        Args:
            x: Input tensor of shape (dispatch_group_size, seq_len, hidden_dim)
            weights: Router weights of shape (num_dispatch_groups, dispatch_group_size, seq_len, num_experts_per_tok) or
                     (dispatch_group_size, seq_len, num_experts_per_tok)
            indices: Expert indices of shape (dispatch_group_size, seq_len, num_experts_per_tok)
            expert_offsets: Base offset for each expert from each chip
                Shape: (dispatch_group_size, num_routed_experts) - from get_gate_outputs()

        Returns:
            If num_dispatch_groups == 1:
                dispatched: shape (dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
                metadata: shape (dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
            If num_dispatch_groups > 1:
                dispatched: shape (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
                metadata: shape (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
        """
        logger.info(f"[TorchDispatchModule.forward] INPUT SHAPES:")
        logger.info(f"  x.shape={x.shape}")
        logger.info(f"  weights.shape={weights.shape}")
        logger.info(f"  indices.shape={indices.shape}")
        logger.info(f"  expert_offsets.shape={expert_offsets.shape}")
        logger.info(f"[TorchDispatchModule.forward] CONFIG:")
        logger.info(f"  dispatch_group_size={self.dispatch_group_size}, experts_per_chip={self.experts_per_chip}")
        logger.info(f"  num_routed_experts={self.num_routed_experts}, num_experts_per_tok={self.num_experts_per_tok}")
        logger.info(
            f"  metadata_len={self.metadata_len}, max_dispatched_tokens_per_expert={self.max_dispatched_tokens_per_expert}"
        )
        logger.info(f"  num_dispatch_groups={self.num_dispatch_groups}")

        assert (
            self.dispatch_group_size == x.shape[0] == weights.shape[0] == indices.shape[0]
        ), f"Mismatched dispatch_group_size across inputs. Expected {self.dispatch_group_size}, got {x.shape[0]}, {weights.shape[0]}, {indices.shape[0]}"
        assert (
            self.seq_len_per_chip == x.shape[1] == weights.shape[1] == indices.shape[1]
        ), f"Mismatched seq_len_per_chip across inputs. Expected {self.seq_len_per_chip}, got {x.shape[1]}, {weights.shape[1]}, {indices.shape[1]}"
        assert (
            self.num_experts_per_tok == indices.shape[-1]
        ), f"Last dimension of indices must match num_experts_per_tok {self.num_experts_per_tok}, got {indices.shape[-1]}"

        dispatched_buffer = torch.zeros(self.dispatched_shape, dtype=torch.float32)
        dispatched_metadata = torch.ones(self.dispatched_metadata_shape, dtype=torch.int32) * -1

        for group in range(self.num_dispatch_groups):
            # Make a mutable copy of offsets for dispatch loop
            offset_copy = expert_offsets.clone()

            # Dispatch tokens and metadata
            for chip in range(self.dispatch_group_size):
                for token in range(self.seq_len_per_chip):
                    for topk_idx in range(self.num_experts_per_tok):
                        routed_expert = indices[chip, token, topk_idx]

                        # Use dispatch table if available, otherwise fall back to division
                        if self.expert_dispatch_table is not None:
                            expert_chip = self.expert_dispatch_table[group, routed_expert].item()
                            if expert_chip == -1:
                                continue  # Expert not present in this dispatch group, skip
                        else:
                            assert (
                                False
                            ), "Dispatch table must be provided in multi-group configuration to determine expert chip mapping"

                        expert_index_within_chip = routed_expert % self.experts_per_chip
                        dst_index = offset_copy[chip, routed_expert]

                        dispatched_buffer[group, expert_chip, expert_index_within_chip, dst_index] = x[chip, token]
                        dispatched_metadata[group, expert_chip, expert_index_within_chip, dst_index] = torch.tensor(
                            [
                                chip,
                                token,
                                topk_idx,
                                routed_expert,
                                torch.tensor(weights[chip, token, topk_idx].item(), dtype=torch.bfloat16)
                                .view(torch.int16)
                                .item(),
                            ]
                            + [0] * (self.metadata_len - 5),
                            dtype=dispatched_metadata.dtype,
                        )
                        offset_copy[chip, routed_expert] += 1

        logger.info(f"[TorchDispatchModule.forward] OUTPUT SHAPES:")
        logger.info(f"  {dispatched_buffer.shape=}")
        logger.info(f"  {dispatched_metadata.shape=}")
        return dispatched_buffer, dispatched_metadata
