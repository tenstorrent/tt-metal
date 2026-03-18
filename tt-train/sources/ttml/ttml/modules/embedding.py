# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding layer for ttml models."""

from __future__ import annotations

import ttnn
import ttml

from .module_base import AbstractModuleBase
from .parameter import Parameter


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
        weight_shape = (1, 1, num_embeddings, embedding_dim)
        if zero_init:
            device = ttml.autograd.AutoContext.get_instance().get_device()
            weight_ttnn = ttnn.zeros(
                weight_shape,
                device=device,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.Layout.TILE,
            )
            self.weight = Parameter(ttml.autograd.create_tensor(weight_ttnn))
        else:
            self.weight = Parameter(
                ttml.ops.randn(
                    weight_shape, dtype=ttnn.DataType.BFLOAT16, mean=0.0, std=0.02
                )
            )

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of embedding layer.

        Args:
            x: Input tensor of token indices, shape [batch_size, 1, 1, seq_len]

        Returns:
            Output tensor of embeddings
        """
        return ttml.ops.embedding.embedding(x, self.weight.tensor)
