# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Post-Combine Reduction Module (PyTorch Reference Implementation)

This module implements the reduction operation after MoE combine.
It sums over the sparse topk dimension (weighted by gate scores).

After MoE combine, each chip has tensor [seq_len, topk, emb_dim]
where each topk slot contains the contribution from one routed expert.

This module:
1. Optional weight multiplication (for weighted MoE sum)
2. Sums over topk dimension (local operation per chip)

Note: The TTNN implementation additionally does reduce-scatter along the TP axis,
but when the tensor is reconstructed for comparison, the effect cancels out.
The torch reference only needs to sum over topk since each row's data matches
the reconstructed TTNN row data.
"""

from typing import Optional

import torch


class TorchReduceModule(torch.nn.Module):
    """Reference implementation for post-combine reduction (sum over topk)."""

    def __init__(self, topk_dim: int = 2):
        """
        Initialize reduce module.

        Args:
            topk_dim: Dimension of the topk axis in the full input tensor
                      (including the leading num_chips dimension).
                      For input [num_chips, seq_len, topk, emb_dim], topk_dim=2.
        """
        super().__init__()
        self.topk_dim = topk_dim

    def forward(
        self,
        combine_output: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reduce combine output by applying weights and summing over topk.

        Args:
            combine_output: [num_chips, seq_len, topk, emb_dim]
            weights: Optional gate weights [num_chips, seq_len, topk]
                     If provided, applies weighted sum: weights * combine_output

        Returns:
            Reduced output: [num_chips, seq_len, emb_dim]
        """
        # 0. Apply weights if provided
        if weights is not None:
            # Broadcast weights: [num_chips, seq_len, topk] -> [num_chips, seq_len, topk, emb_dim]
            combine_output = combine_output * weights.unsqueeze(-1)

        # 1. Sum over topk dimension (collapse expert contributions)
        # [num_chips, seq_len, topk, emb_dim] -> [num_chips, seq_len, emb_dim]
        reduced = combine_output.sum(dim=self.topk_dim)

        return reduced
