import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc


def test_sdpa_causal_baseline(device):
    """Test that causal SDPA still works after host-side changes."""
    torch.manual_seed(42)

    B, NH, S, DH = 1, 8, 256, 64
    q_chunk, k_chunk = 32, 32

    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

    # PyTorch reference (causal)
    scale = 1.0 / (DH**0.5)
    mask = torch.tril(torch.ones(S, S))
    attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale + (1 - mask) * -1e9, dim=-1)
    expected = torch.matmul(attn, V)

    # TTNN causal
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
    )

    Q_tt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    K_tt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    V_tt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tt = ttnn.transformer.scaled_dot_product_attention(
        Q_tt,
        K_tt,
        V_tt,
        is_causal=True,  # Test causal mode
        program_config=program_config,
    )

    output = ttnn.to_torch(output_tt)
    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    assert passing, f"Causal baseline PCC check failed: {pcc}"


def test_sdpa_non_causal_simple(device):
    """Test non-causal SDPA with simplest possible config."""
    torch.manual_seed(42)

    B, NH, S, DH = 1, 1, 64, 64  # Minimal config
    q_chunk, k_chunk = 32, 32

    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

    # PyTorch reference
    scale = 1.0 / (DH**0.5)
    attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
    expected = torch.matmul(attn, V)

    # TTNN non-causal
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1),  # Single core
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        enable_kv_chain_forwarding=False,  # Disable chain forwarding
    )

    Q_tt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    K_tt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    V_tt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tt = ttnn.transformer.scaled_dot_product_attention(
        Q_tt,
        K_tt,
        V_tt,
        is_causal=False,
        program_config=program_config,
    )

    output = ttnn.to_torch(output_tt)
    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    assert passing, f"Non-causal simple PCC check failed: {pcc}"
