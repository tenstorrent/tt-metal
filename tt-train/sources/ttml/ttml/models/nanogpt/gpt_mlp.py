# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GPT-style MLP (feed-forward) layer for NanoGPT."""

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter, RunMode


class GPTMLP(AbstractModuleBase):
    """GPT-style MLP (feed-forward) layer."""

    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        """Initialize GPT MLP layer.

        Args:
            embedding_dim: Dimension of embeddings
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout
        # Note: RunMode is managed by AbstractModuleBase (defaults to TRAIN)

        # First linear: embedding_dim -> embedding_dim * 4
        # Linear weights must be in TILE layout
        fc1_shape = (1, 1, embedding_dim * 4, embedding_dim)
        fc1_np = np.random.normal(0.0, 0.02, size=fc1_shape).astype(ml_dtypes.bfloat16)
        fc1_tensor = ttml.autograd.Tensor.from_numpy(fc1_np, layout=ttnn.Layout.TILE)
        self.fc1 = Parameter(fc1_tensor)

        # Second linear: embedding_dim * 4 -> embedding_dim
        fc2_shape = (1, 1, embedding_dim, embedding_dim * 4)
        fc2_np = np.random.normal(0.0, 0.02, size=fc2_shape).astype(ml_dtypes.bfloat16)
        fc2_tensor = ttml.autograd.Tensor.from_numpy(fc2_np, layout=ttnn.Layout.TILE)
        self.fc2 = Parameter(fc2_tensor)

    # train() and eval() are inherited from AbstractModuleBase

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor after MLP
        """
        # First linear + GELU
        x = ttml.ops.linear.linear_op(x, self.fc1.tensor, None)
        x = ttml.ops.unary.gelu(x)

        # Second linear
        x = ttml.ops.linear.linear_op(x, self.fc2.tensor, None)

        # Dropout (only in training mode, using RunMode from AbstractModuleBase)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            x = ttml.ops.dropout.dropout(x, self.dropout_prob)

        return x

    def __call__(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Call the forward method."""
        return self.forward(x)
