# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding layer for NanoGPT."""

import numpy as np
import ml_dtypes

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
        # Weight must be in TILE layout because embedding_op calls untilize on it
        weight_shape = (1, 1, num_embeddings, embedding_dim)
        weight_np = np.random.normal(0.0, 0.02, size=weight_shape).astype(
            ml_dtypes.bfloat16
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            weight_np, layout=ttml.Layout.TILE
        )
        self.weight = Parameter(weight_tensor)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of embedding layer.

        Args:
            x: Input tensor of token indices, shape [batch_size, 1, 1, seq_len]

        Returns:
            Output tensor of embeddings
        """
        return ttml.ops.embedding.embedding_op(x, self.weight.tensor)

    def __call__(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Call the forward method."""
        return self.forward(x)
