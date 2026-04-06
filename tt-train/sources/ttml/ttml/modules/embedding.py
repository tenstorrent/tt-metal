# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding layer for ttml models."""

from __future__ import annotations

import ttml

from .module_base import AbstractModuleBase
from .parameter import Parameter


class Embedding(AbstractModuleBase):
    """Embedding layer implemented in Python using ttml operations."""

    def __init__(self, num_embeddings: int, embedding_dim: int, weight_init=None) -> None:
        """Initialize embedding layer.

        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of embeddings
            weight_init: Initializer for weight tensor. Defaults to normal(0, 0.02).
        """
        super().__init__()

        if weight_init is None:
            weight_init = ttml.init.normal(0.0, 0.02)

        weight_shape = (1, 1, num_embeddings, embedding_dim)
        self.weight = Parameter(weight_init(weight_shape))

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of embedding layer.

        Args:
            x: Input tensor of token indices, shape [batch_size, 1, 1, seq_len]

        Returns:
            Output tensor of embeddings
        """
        return ttml.ops.embedding.embedding(x, self.weight.tensor)
