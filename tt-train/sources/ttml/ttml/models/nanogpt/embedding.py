# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding layer for NanoGPT."""

from __future__ import annotations

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter


class Embedding(AbstractModuleBase):
    """Embedding layer implemented in Python using ttml operations."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, zero_init: bool = False
    ) -> None:
        """Initialize embedding layer.

        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of embeddings
            zero_init: If True, initialize weights to zero
        """
        super().__init__()

        # Initialize weight tensor: shape [1, 1, num_embeddings, embedding_dim]
        # Weight must be in TILE layout because embedding calls untilize on it
        device = ttml.autograd.AutoContext.get_instance().get_device()
        weight_shape = (1, 1, num_embeddings, embedding_dim)
        if zero_init:
            weight_ttnn = ttnn.zeros(
                weight_shape,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            weight_ttnn = ttml.ops.randn(
                weight_shape, device=device, dtype=ttnn.bfloat16, mean=0.0, std=0.02
            )
        self.weight = Parameter(ttml.autograd.create_tensor(weight_ttnn))

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of embedding layer.

        Args:
            x: Input tensor of token indices, shape [batch_size, 1, 1, seq_len]

        Returns:
            Output tensor of embeddings
        """
        return ttml.ops.embedding.embedding(x, self.weight.tensor)
