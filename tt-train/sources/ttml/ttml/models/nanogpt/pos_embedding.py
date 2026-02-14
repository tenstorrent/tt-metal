# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Positional embedding layer for NanoGPT."""

from __future__ import annotations

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter, RunMode


def sin_pos_embedding_np(seq_len: int, d_model: int) -> np.ndarray:
    """
    Create sinusoidal positional embeddings as described in "Attention Is All You Need".

    Args:
        seq_len: Maximum sequence length
        d_model: Embedding dimension (must be even)

    Returns:
        Positional embeddings of shape (seq_len, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model/2,)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # even indices
    pe[:, 1::2] = np.cos(position * div_term)  # odd indices

    return pe


class PositionalEmbedding(AbstractModuleBase):
    """Positional embedding matching C++ PositionalEmbedding."""

    def __init__(self, sequence_length: int, embedding_dim: int, dropout_prob: float = 0.0) -> None:
        """Initialize positional embedding.

        Args:
            sequence_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            dropout_prob: Dropout probability
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.dropout_prob = dropout_prob

        device = ttml.autograd.AutoContext.get_instance().get_device()
        emb_np = sin_pos_embedding_np(sequence_length, embedding_dim)
        emb_np = emb_np.astype(np.float32)
        emb_np = emb_np.reshape(1, 1, sequence_length, embedding_dim)
        # TODO: Migrate to autograd tensor after branch pruning optimization.
        emb = ttnn.Tensor(emb_np, ttnn.float32, device, ttnn.TILE_LAYOUT)
        emb = ttnn.typecast(emb, ttnn.bfloat16)
        self.positional_embedding = emb

    def forward(self, input: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass: add positional embeddings and apply dropout.

        Args:
            input: Input tensor (token embeddings), shape [batch, 1, seq_len, embedding_dim]

        Returns:
            Output tensor with positional embeddings added
        """
        # Simply add the positional embedding tensor (matching C++ ops::add(input, m_positional_embedding))
        if len(input.shape()) != 4:
            raise ValueError(f"PositionalEmbedding: input tensor must have 4 dimensions. Got rank {len(input.shape())}")
        if input.shape()[2] != self.sequence_length:
            raise ValueError(
                f"PositionalEmbedding: input tensor sequence length ({input.shape()[2]}) does not match the expected value ({self.sequence_length})"
            )
        x = ttml.ops.binary.add(input, self.positional_embedding)
        # Note: It's better to just use Dropout module here
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            x = ttml.ops.dropout.dropout(x, self.dropout_prob)
        return x

    def __call__(self, input: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Call the forward method."""
        return self.forward(input)


class TrainablePositionalEmbedding(AbstractModuleBase):
    """Trainable positional embedding matching C++ TrainablePositionalEmbedding."""

    def __init__(self, sequence_length: int, embedding_dim: int, dropout_prob: float = 0.0) -> None:
        """Initialize trainable positional embedding.

        Args:
            sequence_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            dropout_prob: Dropout probability
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.dropout_prob = dropout_prob

        weight_shape = (1, 1, sequence_length, embedding_dim)
        weight_np = np.random.normal(0.0, 0.02, size=weight_shape).astype(ml_dtypes.bfloat16)
        weight_tensor = ttml.autograd.Tensor.from_numpy(weight_np, layout=ttnn.Layout.TILE)
        self.weight = Parameter(weight_tensor)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass: add positional embeddings and apply dropout.

        Args:
            x: Input tensor (token embeddings), shape [batch, 1, seq_len, embedding_dim]

        Returns:
            Output tensor with positional embeddings added
        """
        # Simply add the positional weight tensor (matching C++ ops::add(input, m_weight))
        if len(x.shape()) != 4:
            raise ValueError(
                f"TrainablePositionalEmbedding: input tensor must have 4 dimensions. Got rank {len(x.shape())}"
            )
        if x.shape()[2] != self.sequence_length:
            raise ValueError(
                f"TrainablePositionalEmbedding: input tensor sequence length ({x.shape()[2]}) does not match the expected value ({self.sequence_length})"
            )
        out = ttml.ops.binary.add(x, self.weight.tensor)
        # Note: It's better to just use Dropout module here
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)
        return out

    def __call__(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Call the forward method."""
        return self.forward(x)
