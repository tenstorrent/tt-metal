# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.utility_functions import comp_pcc


def test_baseline_8x8_grid(device):
    """Test baseline SDPA with 8x8 grid (no chain forwarding - baseline code)"""
    B, NH, S, DH = 1, 1, 64, 64

    torch.manual_seed(42)
    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=32,
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

    # PyTorch reference
    scale = 1.0 / (DH**0.5)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = torch.matmul(attn, V)

    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    assert passing, f"PCC check failed: {pcc}"
    print(f"✓ Baseline test passed with 8x8 grid, PCC={pcc}")


def test_baseline_full_grid(device):
    """Test baseline SDPA with full device grid"""
    B, NH, S, DH = 1, 1, 64, 64

    torch.manual_seed(42)
    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

    grid = device.compute_with_storage_grid_size()
    print(f"Testing with full grid: {grid} ({grid.x}x{grid.y} = {grid.x * grid.y} cores)")

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=32,
        k_chunk_size=32,
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

    # PyTorch reference
    scale = 1.0 / (DH**0.5)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = torch.matmul(attn, V)

    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    assert passing, f"PCC check failed: {pcc}"
    print(f"✓ Baseline test passed with full grid, PCC={pcc}")
