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
        num_chips: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
    ):
        """
        Initialize combine module with configuration parameters.

        Args:
            num_chips: Number of chips in the system
            experts_per_chip: Number of experts per chip
            num_experts_per_tok: Number of experts each token is routed to
        """
        super().__init__()
        self.num_chips = num_chips
        self.experts_per_chip = experts_per_chip
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip

    def forward(
        self,
        dispatched: torch.Tensor,
        metadata: torch.Tensor,
        experts_counter: torch.Tensor,
    ):
        """
        Combine expert outputs back to original token positions.

        Args:
            dispatched: Dispatched tokens of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
            metadata: Metadata tensor containing token positions
            experts_counter: Counter tracking tokens per expert
            seq_len: Sequence length per chip (used for output shape)

        Returns:
            y: Combined output tensor of shape (num_chips, seq_len, num_experts_per_tok, hidden_dim)
        """
        # Infer hidden_dim from dispatched tensor shape
        hidden_dim = dispatched.shape[-1]

        y = torch.zeros(
            (self.num_chips, self.seq_len_per_chip, self.num_experts_per_tok, hidden_dim), dtype=torch.bfloat16
        )

        for chips in range(self.num_chips):
            for experts in range(self.experts_per_chip):
                for i in range(experts_counter[chips, experts]):
                    chip = int(metadata[chips, experts, i, 0])
                    token = int(metadata[chips, experts, i, 1])
                    topk_indice = int(metadata[chips, experts, i, 2])
                    y[chip, token, topk_indice] = dispatched[chips, experts, i]

        return y
