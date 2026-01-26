# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding modules for roofline modeling.

This module provides MockEmbedding and MockTrainablePositionalEmbedding
for roofline estimation of embedding operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockEmbeddingOp, MockAddOp, MockDropoutOp
from .module import MockModule, MockParameter

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockEmbedding(MockModule):
    """Embedding module for roofline estimation.

    Mirrors ttml.models.nanogpt.Embedding for roofline modeling.

    Example:
        >>> emb = MockEmbedding(50304, 768)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> indices = MockTensor((1, 1, 1, 1024), dtype=DataType.INT32)
        >>> output = emb(ctx, indices)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize embedding module.

        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Dimension of embeddings
            dtype: Data type for parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Weight: [1, 1, num_embeddings, embedding_dim]
        self.weight = MockParameter(
            MockTensor((1, 1, num_embeddings, embedding_dim), dtype=dtype)
        )

    def forward(self, ctx: "RooflineContext", indices: MockTensor) -> MockTensor:
        """Forward pass: lookup embeddings.

        Args:
            ctx: Roofline context for estimates
            indices: Token indices [batch, 1, 1, seq_len]

        Returns:
            Embeddings tensor [batch, 1, seq_len, embedding_dim]
        """
        return MockEmbeddingOp.apply(ctx, indices, self.weight.tensor)

    def __repr__(self) -> str:
        return (
            f"MockEmbedding(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim})"
        )


class MockTrainablePositionalEmbedding(MockModule):
    """Trainable positional embedding module for roofline estimation.

    Mirrors ttml.models.nanogpt.TrainablePositionalEmbedding for roofline modeling.
    Adds learnable positional embeddings to input and optionally applies dropout.

    Example:
        >>> pos_emb = MockTrainablePositionalEmbedding(1024, 768, dropout=0.1)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 768))
        >>> output = pos_emb(ctx, x)
    """

    def __init__(
        self,
        sequence_length: int,
        embedding_dim: int,
        dropout: float = 0.0,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize trainable positional embedding.

        Args:
            sequence_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            dropout: Dropout probability
            dtype: Data type for parameters
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        # Weight: [1, 1, sequence_length, embedding_dim]
        self.weight = MockParameter(
            MockTensor((1, 1, sequence_length, embedding_dim), dtype=dtype)
        )

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Forward pass: add positional embeddings and optionally apply dropout.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [batch, 1, seq_len, embedding_dim]

        Returns:
            Output tensor with positional embeddings added
        """
        # Add positional embeddings
        out = MockAddOp.apply(ctx, x, self.weight.tensor)

        # Apply dropout if probability > 0
        if self.dropout_prob > 0:
            out = MockDropoutOp.apply(ctx, out, self.dropout_prob)

        return out

    def __repr__(self) -> str:
        return (
            f"MockTrainablePositionalEmbedding(sequence_length={self.sequence_length}, "
            f"embedding_dim={self.embedding_dim}, dropout={self.dropout_prob})"
        )
