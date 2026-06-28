# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug test for scaled_dot_product_attention hang at Phase 11.

Single tile (1,1,32,32), no mask, auto scale.
B_q_t=1, B_kv_t=1, D_t=1, num_score_tiles=1, num_o_tiles=1.
1 Q-block, 1 KV-block.
"""

import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from .reference import flash_attention_reference


def test_sdpa_debug_single_tile(device):
    """Minimal single-tile test to trace Phase 11 hang."""
    torch.manual_seed(42)

    Q = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    K = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    V = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)

    expected = flash_attention_reference(Q, K, V)

    ttnn_Q = ttnn.from_torch(
        Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)

    assert list(output.shape) == [1, 1, 32, 32]
    torch_output = ttnn.to_torch(output)
    print(f"Output (first 4x4): {torch_output[0,0,:4,:4]}")
    print(f"Expected (first 4x4): {expected[0,0,:4,:4]}")
