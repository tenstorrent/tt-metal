# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "B, NH, S, DH, q_chunk, k_chunk",
    [
        (1, 10, 512, 128, 64, 128),  # Moderate size
        (1, 10, 1024, 128, 64, 128),  # Larger
    ],
    ids=["512seq", "1024seq"],
)
def test_moderate_sizes(device, B, NH, S, DH, q_chunk, k_chunk):
    """Test progressively larger sizes to find where it breaks"""
    torch.manual_seed(1234)

    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        enable_kv_chain_forwarding=False,  # Test without chain first
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

    output = ttnn.to_torch(output_tt)[:, :, :S, :]

    # PyTorch reference
    scale = 1.0 / (DH**0.5)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = torch.matmul(attn, V)

    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    assert passing, f"PCC check failed: {pcc}"
