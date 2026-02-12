# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding layer for NanoGPT."""

from __future__ import annotations

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter


class Embedding(AbstractModuleBase):
    """Embedding layer implemented in Python using ttml operations."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        """Initialize embedding layer.

        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of embeddings
        """
        super().__init__()

        # Initialize weight tensor: shape [1, 1, num_embeddings, embedding_dim]
        # Embedding weights must be BFLOAT16 - use ml_dtypes.bfloat16 on NumPy side
        # Weight must be in TILE layout because embedding calls untilize on it
        weight_shape = (1, 1, num_embeddings, embedding_dim)
        weight_np = np.random.normal(0.0, 0.02, size=weight_shape).astype(
            ml_dtypes.bfloat16
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            weight_np, layout=ttnn.Layout.TILE
        )
        self.weight = Parameter(weight_tensor)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of embedding layer.

        Args:
            x: Input tensor of token indices, shape [batch_size, 1, 1, seq_len]

        Returns:
            Output tensor of embeddings
        """
        return ttml.ops.embedding.embedding(x, self.weight.tensor)
