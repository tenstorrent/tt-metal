# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding layer for Llama."""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, Parameter
from ttml.modules.parameter import TensorMetadata


class Embedding(AbstractModuleBase):
    """Embedding layer implemented in Python using ttml operations."""

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        """Initialize embedding layer.

        Args:
            num_embeddings: Size of vocabulary.
            embedding_dim: Dimension of embeddings.
            **kwargs: Forwarded to AbstractModuleBase (mesh_device, tp_plan, etc.).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Match C++ modules/embedding_module.cpp: normal_init(..., {0.F, 1.F})
        self.weight = Parameter(
            TensorMetadata(
                shape=(1, 1, num_embeddings, embedding_dim),
                init_fn=ttml.init.normal(0.0, 1.0),
            )
        )

        super().__init__(**kwargs)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of embedding layer.

        Args:
            x: Input tensor of token indices, shape [batch_size, 1, 1, seq_len]

        Returns:
            Output tensor of embeddings
        """
        return ttml.ops.embedding.embedding(x, self.weight.tensor)
