# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for Informer embedding components."""

import math

import pytest
import torch

import ttnn
from models.demos.informer.reference.torch_reference import compute_metrics
from models.demos.informer.tt.embeddings import PositionalEmbedding, TemporalEmbedding, ValueEmbedding
from models.demos.informer.tt.ops import sinusoidal_position_encoding, to_torch


class TestPositionalEmbedding:
    """Test sinusoidal positional encoding."""

    @pytest.mark.parametrize("seq_len", [32, 64, 96])
    @pytest.mark.parametrize("d_model", [64, 128, 256])
    def test_positional_encoding_pcc(self, device, seq_len, d_model):
        """Test positional encoding matches PyTorch reference."""
        # PyTorch reference
        position = torch.arange(seq_len, dtype=torch.float32)[:, None]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe_expected = torch.zeros((seq_len, d_model), dtype=torch.float32)
        pe_expected[:, 0::2] = torch.sin(position * div_term)
        pe_expected[:, 1::2] = torch.cos(position * div_term)

        # TTNN implementation
        dtype = ttnn.bfloat16
        pe_ttnn = sinusoidal_position_encoding(seq_len, d_model, device=device, dtype=dtype)
        pe_actual = to_torch(pe_ttnn).squeeze(0)

        pcc = compute_metrics(pe_expected, pe_actual)[2]
        assert pcc > 0.99, f"Positional encoding PCC {pcc:.4f} < 0.99"


class TestValueEmbedding:
    """Test value embedding layer."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [32, 64])
    @pytest.mark.parametrize(
        "input_dim,d_model",
        [
            (7, 64),
            (4, 128),
        ],
    )
    def test_value_embedding_pcc(self, device, batch_size, seq_len, input_dim, d_model):
        """Test value embedding matches PyTorch reference."""
        rng = torch.Generator().manual_seed(42)
        dtype = ttnn.bfloat16

        # Create TTNN embedding
        ttnn_emb = ValueEmbedding(input_dim, d_model, rng, device=device, dtype=dtype)

        # Input
        x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # PyTorch reference using same weights
        expected = x @ ttnn_emb.weight_torch.T + ttnn_emb.bias_torch

        # TTNN forward
        actual = to_torch(ttnn_emb(x_ttnn))

        pcc = compute_metrics(expected, actual)[2]
        assert pcc > 0.98, f"Value embedding PCC {pcc:.4f} < 0.98"


class TestTemporalEmbedding:
    """Test temporal embedding layer."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 64])
    @pytest.mark.parametrize(
        "time_dim,d_model",
        [
            (4, 64),
            (6, 128),
        ],
    )
    def test_temporal_embedding_pcc(self, device, batch_size, seq_len, time_dim, d_model):
        """Test temporal embedding matches PyTorch reference."""
        rng = torch.Generator().manual_seed(42)
        dtype = ttnn.bfloat16

        # Create TTNN embedding
        ttnn_emb = TemporalEmbedding(time_dim, d_model, rng, device=device, dtype=dtype)

        # Input
        x = torch.randn(batch_size, seq_len, time_dim, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # PyTorch reference using same weights
        expected = x @ ttnn_emb.weight_torch.T + ttnn_emb.bias_torch

        # TTNN forward
        actual = to_torch(ttnn_emb(x_ttnn))

        pcc = compute_metrics(expected, actual)[2]
        assert pcc > 0.98, f"Temporal embedding PCC {pcc:.4f} < 0.98"


class TestPositionalEmbeddingClass:
    """Test PositionalEmbedding class."""

    @pytest.mark.parametrize("max_len", [128, 256])
    @pytest.mark.parametrize("d_model", [64, 128])
    def test_positional_embedding_class(self, device, max_len, d_model):
        """Test PositionalEmbedding class returns correct shape and values."""
        dtype = ttnn.bfloat16
        pos_emb = PositionalEmbedding(max_len, d_model, device=device, dtype=dtype)

        # Test different lengths
        for length in [32, 64, max_len]:
            pe = pos_emb(length)
            pe_torch = to_torch(pe)

            assert pe_torch.shape == (1, length, d_model), f"Expected (1, {length}, {d_model}), got {pe_torch.shape}"
            assert torch.isfinite(pe_torch).all(), "Positional encoding contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
