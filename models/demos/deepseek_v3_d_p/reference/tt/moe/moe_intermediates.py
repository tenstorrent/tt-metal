"""
Data structures for Torch MoE intermediate values.
"""

from dataclasses import dataclass

import torch


@dataclass
class MoEIntermediates:
    """Data structure holding intermediate values from MoE forward pass for debugging."""

    dispatched_buffer: (
        torch.Tensor
    )  # (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, hidden_dim)
    metadata: torch.Tensor  # (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, metadata_len)
    expert_outputs: torch.Tensor  # Same shape as dispatched_buffer
    shared_output: torch.Tensor  # (dispatch_group_size, seq_len_per_chip, hidden_dim)
    combined_output: torch.Tensor  # (dispatch_group_size, seq_len_per_chip, num_experts_per_tok, hidden_dim)
    routed_output: torch.Tensor  # (dispatch_group_size, seq_len_per_chip, hidden_dim)
