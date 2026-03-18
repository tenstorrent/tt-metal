# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Post-Combine Reduction Module (PyTorch Reference Implementation)

This module implements the reduction operation after MoE combine.
It sums over the sparse topk dimension and reduces across chips.

After MoE combine, each chip has sparse tensor [seq_len, topk, hidden_dim]
where only positions for local experts have valid data.

This module:
1. Optional weight multiplication (for weighted MoE sum)
2. Sums over topk dimension (local operation)
3. Sums across chips in the same row (distributed reduction)
4. Returns TP-sharded output ready for next layer
"""

from typing import List, Optional

import torch


class TorchReduceModule(torch.nn.Module):
    """Reference implementation for post-combine reduction."""

    def __init__(self, num_reduce_chips: int, topk_dim: int = 1):
        """
        Initialize reduce module.

        Args:
            num_reduce_chips: Number of chips to reduce across (e.g., 2 for 4x2 mesh columns)
            topk_dim: Dimension of the topk axis in input tensor
        """
        super().__init__()
        self.num_reduce_chips = num_reduce_chips
        self.topk_dim = topk_dim

    def forward(
        self,
        combine_output: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Reduce combine output by summing topk and reducing across chips.

        Args:
            combine_output: [num_chips, seq_len, topk, hidden_dim] (sparse in topk)
            weights: Optional gate weights [num_chips, seq_len, topk]
                     If provided, applies weighted sum: weights * combine_output

        Returns:
            List of per-chip shards: each [seq_len, hidden_dim // num_reduce_chips]
        """
        # 0. Apply weights if provided
        if weights is not None:
            # Broadcast weights: [num_chips, seq_len, topk] -> [num_chips, seq_len, topk, hidden_dim]
            combine_output = combine_output * weights.unsqueeze(-1)

        # 1. Sum over topk dimension (collapse sparse contributions)
        # [num_chips, seq_len, topk, hidden_dim] -> [num_chips, seq_len, hidden_dim]
        summed = combine_output.sum(dim=self.topk_dim + 1)  # +1 because of leading chips dim

        # 2. Sum across chips (reduce) - simulates reduce operation
        # [num_chips, seq_len, hidden_dim] -> [seq_len, hidden_dim]
        reduced = summed.sum(dim=0)

        # 3. Scatter into shards (one per chip)
        # Each chip gets [seq_len, hidden_dim // num_reduce_chips]
        shard_size = reduced.shape[-1] // self.num_reduce_chips
        shards = [reduced[..., i * shard_size : (i + 1) * shard_size] for i in range(self.num_reduce_chips)]

        return shards
