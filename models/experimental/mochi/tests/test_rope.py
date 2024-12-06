import pytest
import torch
import ttnn
from models.experimental.mochi.common import compute_metrics

from genmo.mochi_preview.dit.joint_model.temporal_rope import apply_rotary_emb_qk_real


def test_apply_rotary_emb_qk_real():
    # Test parameters
    batch_size = 2
    seq_len = 4
    num_heads = 8
    head_dim = 64

    # Create input tensors
    xqk = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cos = torch.randn(seq_len, num_heads, head_dim // 2)
    freqs_sin = torch.randn(seq_len, num_heads, head_dim // 2)

    # Run reference implementation
    gt = apply_rotary_emb_qk_real(xqk, freqs_cos, freqs_sin)

    # xqk_even = xqk[..., 0::2]
    # xqk_odd = xqk[..., 1::2]
    # cos_part = (xqk_even * freqs_cos - xqk_odd * freqs_sin).type_as(xqk)
    # sin_part = (xqk_even * freqs_sin + xqk_odd * freqs_cos).type_as(xqk)
    # expected_out = torch.stack([cos_part, sin_part], dim=-1).flatten(-2)

    # Run ttnn implementation
    xqk_tt = ttnn.from_torch(xqk)
    freqs_cos_tt = ttnn.from_torch(freqs_cos)
    freqs_sin_tt = ttnn.from_torch(freqs_sin)
    out_tt = ttnn.experimental.rotary_embedding_llama(xqk_tt, freqs_cos_tt, freqs_sin_tt)
    out = ttnn.to_torch(out_tt)

    # Shape assertions
    assert out.shape == xqk.shape

    # Compute accuracy metrics
    pcc, mse, mae = compute_metrics(gt, out)

    # Check if model meets requirements
    pcc_required = 0.99
    assert pcc >= pcc_required, f"Output does not meet PCC requirement {pcc_required}: PCC={pcc}, MSE={mse}, MAE={mae}"


def test_apply_rotary_emb_qk_real_simple():
    pytest.skip()
    """Test with simple values for manual verification"""
    # Simple test case with known values
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a simple 2D input tensor with alternating 1s and 2s
    xqk = torch.tensor([[1, 2, 1, 2], [1, 2, 1, 2]], device=device, dtype=torch.bfloat16)
    xqk = xqk.view(1, 2, 1, 4)  # reshape to (batch=1, seq_len=2, heads=1, dim=4)

    # Create simple frequency tensors
    freqs_cos = torch.tensor([[1, 1], [0, 0]], device=device, dtype=torch.bfloat16)  # (seq_len=2, dim//2=2)
    freqs_sin = torch.tensor([[0, 0], [1, 1]], device=device, dtype=torch.bfloat16)

    # Expected output based on manual calculation
    # For first sequence position (cos=1, sin=0):
    #   even_indices (1,1): 1*1 - 2*0 = 1
    #   odd_indices (2,2): 1*0 + 2*1 = 2
    # For second sequence position (cos=0, sin=1):
    #   even_indices (1,1): 1*0 - 2*1 = -2
    #   odd_indices (2,2): 1*1 + 2*0 = 1
    expected = torch.tensor([[1, 2, 1, 2], [-2, 1, -2, 1]], device=device, dtype=torch.bfloat16)
    expected = expected.view(1, 2, 1, 4)

    # Run ttnn implementation
    xqk_tt = ttnn.from_torch(xqk)
    freqs_cos_tt = ttnn.from_torch(freqs_cos)
    freqs_sin_tt = ttnn.from_torch(freqs_sin)
    out_tt = apply_rotary_emb_qk_real(xqk_tt, freqs_cos_tt, freqs_sin_tt)
    out = ttnn.to_torch(out_tt)

    # Compute accuracy metrics
    pcc, mse, mae = compute_metrics(expected, out)

    # Check if model meets requirements
    pcc_required = 0.99
    assert pcc >= pcc_required, f"Output does not meet PCC requirement {pcc_required}: PCC={pcc}, MSE={mse}, MAE={mae}"
    assert out.dtype == torch.bfloat16


def test_apply_rotary_emb_qk_real_edge_cases():
    pytest.skip()
    """Test edge cases and special values"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with zero input
    xqk_zeros = torch.zeros(1, 1, 1, 4, device=device, dtype=torch.bfloat16)
    freqs_cos = torch.ones(1, 2, device=device, dtype=torch.bfloat16)
    freqs_sin = torch.ones(1, 2, device=device, dtype=torch.bfloat16)

    xqk_zeros_tt = ttnn.from_torch(xqk_zeros)
    freqs_cos_tt = ttnn.from_torch(freqs_cos)
    freqs_sin_tt = ttnn.from_torch(freqs_sin)
    out_zeros = ttnn.to_torch(apply_rotary_emb_qk_real(xqk_zeros_tt, freqs_cos_tt, freqs_sin_tt))

    assert torch.all(out_zeros == 0)

    # Test with ones
    xqk_ones = torch.ones(1, 1, 1, 4, device=device, dtype=torch.bfloat16)
    out_ones = ttnn.to_torch(apply_rotary_emb_qk_real(ttnn.from_torch(xqk_ones), freqs_cos_tt, freqs_sin_tt))

    # For all ones input with cos=1, sin=1:
    # even indices: 1*1 - 1*1 = 0
    # odd indices: 1*1 + 1*1 = 2
    expected_ones = torch.tensor([0, 2, 0, 2], device=device, dtype=torch.bfloat16).view(1, 1, 1, 4)
    torch.testing.assert_close(out_ones, expected_ones, rtol=1e-3, atol=1e-3)


def test_apply_rotary_emb_qk_real_shapes():
    pytest.skip()
    """Test various input shapes"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_shapes = [
        # (batch, seq_len, num_heads, head_dim)
        (1, 1, 1, 64),
        (2, 4, 8, 64),
        (3, 16, 16, 128),
    ]

    for batch, seq_len, num_heads, head_dim in test_shapes:
        xqk = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        freqs_cos = torch.randn(seq_len, head_dim // 2, device=device, dtype=torch.bfloat16)
        freqs_sin = torch.randn(seq_len, head_dim // 2, device=device, dtype=torch.bfloat16)

        xqk_tt = ttnn.from_torch(xqk)
        freqs_cos_tt = ttnn.from_torch(freqs_cos)
        freqs_sin_tt = ttnn.from_torch(freqs_sin)
        out = ttnn.to_torch(apply_rotary_emb_qk_real(xqk_tt, freqs_cos_tt, freqs_sin_tt))

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert out.dtype == torch.bfloat16
