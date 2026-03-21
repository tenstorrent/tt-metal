# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared Embedding module (ttml.modules.Embedding)."""

import numpy as np
import pytest

import ttnn
import ttml
from ttml.modules import Embedding, Parameter


class TestEmbedding:
    """Tests for Embedding layer."""

    def test_embedding_creation(self):
        """Test that Embedding layer can be created."""
        num_embeddings = 100
        embedding_dim = 64

        embedding = Embedding(num_embeddings, embedding_dim)

        assert embedding is not None
        assert hasattr(embedding, "weight")
        assert isinstance(embedding.weight, Parameter)

    def test_embedding_weight_shape(self):
        """Test that embedding weight has correct shape."""
        num_embeddings = 100
        embedding_dim = 64

        embedding = Embedding(num_embeddings, embedding_dim)

        assert embedding.weight.tensor.shape() == [1, 1, num_embeddings, embedding_dim]

    def test_embedding_forward_shape(self):
        """Test that embedding forward pass produces correct shape."""
        num_embeddings = 128  # Tile-aligned
        embedding_dim = 64  # Tile-aligned
        batch_size = 4
        seq_len = 32  # Tile-aligned

        embedding = Embedding(num_embeddings, embedding_dim)

        # Create input token indices
        indices = np.random.randint(
            0, num_embeddings, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            indices, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Forward pass
        output = embedding(input_tensor)
        output_shape = output.shape()

        # Output should be [batch_size, 1, seq_len, embedding_dim] (4D)
        assert len(output_shape) == 4
        assert output_shape[0] == batch_size
        assert output_shape[1] == 1
        assert output_shape[2] == seq_len
        assert output_shape[3] == embedding_dim

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_embedding_backward(self):
        """Test that embedding backward pass computes gradients."""
        num_embeddings = 64  # Tile-aligned
        embedding_dim = 64  # Tile-aligned
        batch_size = 2
        seq_len = 32  # Tile-aligned (must be divisible by tile width)

        embedding = Embedding(num_embeddings, embedding_dim)

        # Create input
        indices = np.random.randint(
            0, num_embeddings, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            indices, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Forward and backward
        output = embedding(input_tensor)
        loss = ttml.ops.unary.mean(output)
        loss.backward(False)

        # Check gradient exists on weight
        assert (
            embedding.weight.tensor.is_grad_initialized()
        ), "Embedding weight should have gradient"

        ttml.autograd.AutoContext.get_instance().reset_graph()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
