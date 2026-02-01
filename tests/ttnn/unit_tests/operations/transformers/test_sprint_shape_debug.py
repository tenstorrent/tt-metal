# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def test_sprint_shape(device):
    """Test the exact shape from the stuck sprint test"""
    B, NH, S, DH = 1, 10, 2368, 128
    q_chunk_size, k_chunk_size = 64, 128

    torch.manual_seed(1234)
    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        enable_kv_chain_forwarding=False,  # TEMP: Disable to test if this is the issue
    )

    Q_tt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    K_tt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    V_tt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    print(f"Running SDPA with B={B}, NH={NH}, S={S}, DH={DH}, q_chunk={q_chunk_size}, k_chunk={k_chunk_size}")
    output_tt = ttnn.transformer.scaled_dot_product_attention(
        Q_tt,
        K_tt,
        V_tt,
        is_causal=False,
        program_config=program_config,
    )
    print("SDPA completed successfully!")
