# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for Informer attention components."""


import pytest
import torch

import ttnn
from models.demos.informer.reference.torch_reference import compute_metrics, torch_mha_reference
from models.demos.informer.tt.attention import MultiHeadAttention
from models.demos.informer.tt.ops import make_causal_mask, to_torch


class TestMultiHeadAttention:
    """Test Multi-Head Attention with standard and ProbSparse variants."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 64])
    @pytest.mark.parametrize("d_model,n_heads", [(64, 2), (128, 4)])
    def test_full_attention_pcc(self, device, batch_size, seq_len, d_model, n_heads):
        """Test standard multi-head attention (non-ProbSparse)."""
        rng = torch.Generator().manual_seed(42)
        dtype = ttnn.bfloat16
        mask_value = -1e4

        # Create TTNN attention
        ttnn_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            rng=rng,
            device=device,
            dtype=dtype,
            prob_sparse=False,
            factor=5,
            mask_value=mask_value,
            use_sdpa=True,
            use_sharded=True,
        )

        # Input tensors
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # Forward pass - self-attention (Q=K=V)
        actual = to_torch(ttnn_attn(x_ttnn, x_ttnn, x_ttnn, None))

        # PyTorch reference
        expected = torch_mha_reference(
            ttnn_attn,
            x,
            x,
            x,
            None,
            prob_sparse=False,
            factor=5,
            mask_value=mask_value,
        )

        pcc = compute_metrics(expected, actual)[2]
        assert pcc > 0.95, f"Full attention PCC {pcc:.4f} < 0.95"

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 64])
    @pytest.mark.parametrize("d_model,n_heads", [(64, 2), (128, 4)])
    def test_prob_sparse_attention_pcc(self, device, batch_size, seq_len, d_model, n_heads):
        """Test ProbSparse multi-head attention."""
        rng = torch.Generator().manual_seed(42)
        dtype = ttnn.bfloat16
        mask_value = -1e4
        factor = 5

        # Create TTNN attention with ProbSparse
        ttnn_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            rng=rng,
            device=device,
            dtype=dtype,
            prob_sparse=True,
            factor=factor,
            mask_value=mask_value,
            use_sdpa=True,
            use_sharded=True,
        )

        # Input tensors
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # Forward pass
        actual = to_torch(ttnn_attn(x_ttnn, x_ttnn, x_ttnn, None, q_valid_len=seq_len, k_valid_len=seq_len))

        # PyTorch reference
        expected = torch_mha_reference(
            ttnn_attn,
            x,
            x,
            x,
            None,
            prob_sparse=True,
            factor=factor,
            mask_value=mask_value,
            q_valid_len=seq_len,
            k_valid_len=seq_len,
        )

        pcc = compute_metrics(expected, actual)[2]
        # ProbSparse may have slightly lower PCC due to top-k selection differences
        assert pcc > 0.90, f"ProbSparse attention PCC {pcc:.4f} < 0.90"

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 64])
    @pytest.mark.parametrize("d_model,n_heads", [(64, 2)])
    def test_causal_attention_pcc(self, device, batch_size, seq_len, d_model, n_heads):
        """Test causal (masked) attention."""
        rng = torch.Generator().manual_seed(42)
        dtype = ttnn.bfloat16
        mask_value = -1e4

        # Create TTNN attention
        ttnn_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            rng=rng,
            device=device,
            dtype=dtype,
            prob_sparse=False,
            factor=5,
            mask_value=mask_value,
            use_sdpa=True,
            use_sharded=True,
        )

        # Input tensors
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        x_ttnn = ttnn.from_torch(x, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # Causal mask
        from models.demos.informer.reference.torch_reference import make_causal_mask as torch_causal_mask

        causal_mask_torch = torch_causal_mask(seq_len, mask_value, device=x.device)
        causal_mask_ttnn = make_causal_mask(
            seq_len, batch=batch_size, heads=n_heads, device=device, dtype=dtype, mask_value=mask_value
        )

        # Forward pass
        actual = to_torch(ttnn_attn(x_ttnn, x_ttnn, x_ttnn, causal_mask_ttnn))

        # PyTorch reference
        expected = torch_mha_reference(
            ttnn_attn,
            x,
            x,
            x,
            causal_mask_torch,
            prob_sparse=False,
            factor=5,
            mask_value=mask_value,
        )

        pcc = compute_metrics(expected, actual)[2]
        assert pcc > 0.95, f"Causal attention PCC {pcc:.4f} < 0.95"


class TestCrossAttention:
    """Test cross-attention between encoder and decoder."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("q_len,k_len", [(32, 64), (64, 48)])
    @pytest.mark.parametrize("d_model,n_heads", [(64, 2)])
    def test_cross_attention_pcc(self, device, batch_size, q_len, k_len, d_model, n_heads):
        """Test cross-attention with different Q and K/V lengths."""
        rng = torch.Generator().manual_seed(42)
        dtype = ttnn.bfloat16
        mask_value = -1e4

        # Create TTNN attention
        ttnn_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            rng=rng,
            device=device,
            dtype=dtype,
            prob_sparse=False,
            factor=5,
            mask_value=mask_value,
            use_sdpa=True,
            use_sharded=True,
        )

        # Input tensors
        q = torch.randn(batch_size, q_len, d_model, dtype=torch.float32)
        kv = torch.randn(batch_size, k_len, d_model, dtype=torch.float32)
        q_ttnn = ttnn.from_torch(q, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        kv_ttnn = ttnn.from_torch(kv, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        # Forward pass
        actual = to_torch(ttnn_attn(q_ttnn, kv_ttnn, kv_ttnn, None))

        # PyTorch reference
        expected = torch_mha_reference(
            ttnn_attn,
            q,
            kv,
            kv,
            None,
            prob_sparse=False,
            factor=5,
            mask_value=mask_value,
        )

        pcc = compute_metrics(expected, actual)[2]
        assert pcc > 0.95, f"Cross attention PCC {pcc:.4f} < 0.95"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
