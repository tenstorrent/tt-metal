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
        num_chips: int,
        experts_per_chip: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        seq_len_per_chip: int,
        hidden_dim: int = 7 * 1024,
        num_ep_ranks: int = 1,
        expert_dispatch_table: torch.Tensor = None,
    ):
        """
        Initialize dispatch module with configuration parameters.

        Args:
            num_chips: Number of chips in the system
            experts_per_chip: Number of experts per chip
            n_routed_experts: Total number of routed experts across all chips
            metadata_len: Length of metadata per token (stores: chip, token, topk_indice, routed_expert, weight)
            max_dispatched_tokens_per_expert: Maximum number of tokens that can be dispatched to each expert
            expert_dispatch_table: Optional dispatch table of shape (num_ep_ranks, n_routed_experts)
                Maps expert ID to logical chip ID in dispatch axis, -1 if not present
        """
        super().__init__()
        self.num_chips = num_chips
        self.experts_per_chip = experts_per_chip
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.metadata_len = metadata_len
        self.max_dispatched_tokens_per_expert = max_dispatched_tokens_per_expert
        self.seq_len_per_chip = seq_len_per_chip
        self.num_ep_ranks = num_ep_ranks
        self.expert_dispatch_table = expert_dispatch_table

        # Oversized buffer (max_dispatched_tokens_per_expert) to simplify dispatch logic
        self.dispatched_shape = (
            num_ep_ranks,
            num_chips,
            self.experts_per_chip,
            self.max_dispatched_tokens_per_expert,
            hidden_dim,
        )
        self.dispatched_metadata_shape = (
            num_ep_ranks,
            num_chips,
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
        chip_to_n_routed_expert_offset: torch.Tensor,
    ):
        """
        Route tokens from their original positions to expert-specific buffers distributed across chips.

        Simulates MoE dispatch: each token is routed to multiple experts based on router indices.
        Tokens are gathered into per-expert buffers with metadata tracking their origin for later recombination.

        Args:
            x: Input tensor of shape (num_chips, seq_len, hidden_dim)
            weights: Router weights of shape (num_ep_ranks, num_chips, seq_len, num_experts_per_tok) or
                     (num_chips, seq_len, num_experts_per_tok)
            indices: Expert indices of shape (num_chips, seq_len, num_experts_per_tok)
            chip_to_n_routed_expert_offset: Base offset for each expert from each chip
                Shape: (num_chips, n_routed_experts) - from get_gate_outputs()

        Returns:
            If num_ep_ranks == 1:
                dispatched: shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
                metadata: shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
            If num_ep_ranks > 1:
                dispatched: shape (num_ep_ranks, num_chips, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
                metadata: shape (num_ep_ranks, num_chips, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
        """
        logger.info(f"[TorchDispatchModule.forward] INPUT SHAPES:")
        logger.info(f"  x.shape={x.shape}")
        logger.info(f"  weights.shape={weights.shape}")
        logger.info(f"  indices.shape={indices.shape}")
        logger.info(f"  chip_to_n_routed_expert_offset.shape={chip_to_n_routed_expert_offset.shape}")
        logger.info(f"[TorchDispatchModule.forward] CONFIG:")
        logger.info(f"  num_chips={self.num_chips}, experts_per_chip={self.experts_per_chip}")
        logger.info(f"  n_routed_experts={self.n_routed_experts}, num_experts_per_tok={self.num_experts_per_tok}")
        logger.info(
            f"  metadata_len={self.metadata_len}, max_dispatched_tokens_per_expert={self.max_dispatched_tokens_per_expert}"
        )
        logger.info(f"  num_ep_ranks={self.num_ep_ranks}")

        assert (
            self.num_chips == x.shape[0] == weights.shape[0] == indices.shape[0]
        ), f"Mismatched num_chips across inputs. Expected {self.num_chips}, got {x.shape[0]}, {weights.shape[0]}, {indices.shape[0]}"
        assert (
            self.seq_len_per_chip == x.shape[1] == weights.shape[1] == indices.shape[1]
        ), f"Mismatched seq_len_per_chip across inputs. Expected {self.seq_len_per_chip}, got {x.shape[1]}, {weights.shape[1]}, {indices.shape[1]}"
        assert (
            self.num_experts_per_tok == indices.shape[-1]
        ), f"Last dimension of indices must match num_experts_per_tok {self.num_experts_per_tok}, got {indices.shape[-1]}"

        dispatched_buffer = torch.zeros(self.dispatched_shape, dtype=torch.float32)
        dispatched_metadata = torch.ones(self.dispatched_metadata_shape, dtype=torch.int32) * -1

        for r in range(self.num_ep_ranks):
            # Make a mutable copy of offsets for dispatch loop
            offset_copy = chip_to_n_routed_expert_offset.clone()

            # Dispatch tokens and metadata
            for chip in range(self.num_chips):
                for token in range(self.seq_len_per_chip):
                    for topk_indice in range(self.num_experts_per_tok):
                        routed_expert = indices[chip, token, topk_indice]

                        # Use dispatch table if available, otherwise fall back to division
                        if self.expert_dispatch_table is not None:
                            expert_chip = self.expert_dispatch_table[r, routed_expert].item()
                            if expert_chip == -1:
                                continue  # Expert not present in this EP rank, skip
                        else:
                            assert (
                                False
                            ), "Dispatch table must be provided in multi-EP rank configuration to determine expert chip mapping"
                            # expert_chip = routed_expert // self.experts_per_chip

                        expert_index_within_chip = routed_expert % self.experts_per_chip
                        dst_index = offset_copy[chip, routed_expert]

                        dispatched_buffer[r, expert_chip, expert_index_within_chip, dst_index] = x[chip, token]
                        dispatched_metadata[r, expert_chip, expert_index_within_chip, dst_index] = torch.tensor(
                            [
                                chip,
                                token,
                                topk_indice,
                                routed_expert,
                                torch.tensor(weights[chip, token, topk_indice].item(), dtype=torch.bfloat16)
                                .view(torch.int16)
                                .item(),
                            ]
                            + [0] * (self.metadata_len - 5),
                            dtype=dispatched_metadata.dtype,
                        )
                        offset_copy[chip, routed_expert] += 1

        # dispatched, metadata = self._dispatch_single_rank(x, weights, indices, chip_to_n_routed_expert_offset)
        logger.info(f"[TorchDispatchModule.forward] OUTPUT SHAPES (single-rank):")
        logger.info(f"  {dispatched_buffer.shape=}")
        logger.info(f"  {dispatched_metadata.shape=}")
        return dispatched_buffer, dispatched_metadata
