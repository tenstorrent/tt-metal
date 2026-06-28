# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug test for scaled_dot_product_attention — single tile, all ones.

Hand-calculated expected values for all-ones (1,1,32,32) input:
- Q = K = V = all ones, D=32, scale=1/sqrt(32)≈0.176777
- QK^T = 32 (sum of 32 ones * 32 ones / 32... actually Q@K^T = sum(Q_i * K_i) = 32)
  Wait: each element of Q@K^T = sum over D of Q[0,d]*K[0,d] = sum of 32 ones = 32
- scores = 32 * 0.176777 = 5.656854
- softmax(scores) = exp(5.657) / exp(5.657) = 1.0 (single element, softmax of single value = 1)
- output = 1.0 * V = V = all ones

So for all-ones input, output should be all ones.
"""
import math
import torch
import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def test_all_ones_single_tile(device):
    """Single tile (1,1,32,32), all ones — should produce all ones output."""
    Q = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
    K = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
    V = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)

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

    torch_output = ttnn.to_torch(output)
    assert list(output.shape) == [1, 1, 32, 32]

    # For all-ones input, output should be all ones
    output_f32 = torch_output.float()
    print(
        f"Output min={output_f32.min().item():.6f} max={output_f32.max().item():.6f} mean={output_f32.mean().item():.6f}"
    )

    assert torch.allclose(
        output_f32, torch.ones_like(output_f32), atol=0.01
    ), f"Expected all ones, got min={output_f32.min().item()} max={output_f32.max().item()}"
