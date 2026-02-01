# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.common.utility_functions import comp_pcc


def test_minimal_with_full_grid(device):
    """
    Minimal test case to debug grid size issues.
    Start with smallest possible shape and full device grid.
    """
    B, NH, S, DH = 1, 1, 64, 64  # Minimal shape
    q_chunk, k_chunk = 32, 32

    print(f"\n=== Test Configuration ===")
    grid = device.compute_with_storage_grid_size()
    print(f"Device grid: {grid} ({grid.x}x{grid.y} = {grid.x * grid.y} cores)")
    print(f"Shape: B={B}, NH={NH}, S={S}, DH={DH}")
    print(f"Chunks: q={q_chunk}, k={k_chunk}")
    print(f"Q chunks: {S // q_chunk}")
    print(f"Total Q chunks: {B * NH * (S // q_chunk)}")
    print(f"=========================\n")

    torch.manual_seed(42)
    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        enable_kv_chain_forwarding=True,  # Test with optimization enabled
    )

    print("Converting to TTNN tensors...")
    Q_tt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    K_tt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    V_tt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    print("Running SDPA...")
    output_tt = ttnn.transformer.scaled_dot_product_attention(
        Q_tt,
        K_tt,
        V_tt,
        is_causal=False,
        program_config=program_config,
    )

    print("Converting back to torch...")
    output = ttnn.to_torch(output_tt)

    # PyTorch reference
    scale = 1.0 / (DH**0.5)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = torch.matmul(attn, V)

    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    print(f"PCC: {pcc}, Passing: {passing}")
    assert passing, f"PCC check failed: {pcc}"
    print("✓ Test passed!")


def test_with_chain_disabled(device):
    """Test same config but with chain disabled to verify non-chain path works."""
    B, NH, S, DH = 1, 1, 64, 64

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=32,
        enable_kv_chain_forwarding=False,  # Disable to test non-chain path
    )

    torch.manual_seed(42)
    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

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

    # Verify
    scale = 1.0 / (DH**0.5)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = torch.matmul(attn, V)

    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    assert passing, f"PCC check failed: {pcc}"
