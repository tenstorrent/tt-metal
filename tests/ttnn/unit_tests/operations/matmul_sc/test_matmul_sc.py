# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
matmul_sc Integration Test

Verifies correctness of the matmul_sc operation against torch.matmul.

Run with:
    scripts/tt-test.sh tests/ttnn/unit_tests/operations/matmul_sc/test_matmul_sc.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul_sc import matmul_sc


@pytest.mark.parametrize(
    "M, K, N",
    [
        pytest.param(32, 32, 32, id="single_tile"),
        pytest.param(64, 64, 128, id="multi_tile"),
        pytest.param(32, 128, 64, id="non_square"),
        pytest.param(32, 256, 32, id="large_k"),
    ],
)
def test_matmul_sc_correctness(device, M, K, N):
    """Test that matmul_sc output matches torch.matmul within bf16 tolerances."""
    torch_a = torch.randn(M, K, dtype=torch.bfloat16)
    torch_b = torch.randn(K, N, dtype=torch.bfloat16)

    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = matmul_sc(ttnn_a, ttnn_b)

    assert list(ttnn_output.shape) == [M, N], f"Shape mismatch: got {list(ttnn_output.shape)}, expected [{M}, {N}]"
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.TILE_LAYOUT

    actual = ttnn.to_torch(ttnn_output)
    expected = torch.matmul(torch_a, torch_b)
    assert torch.allclose(actual, expected, rtol=0.05, atol=0.2), (
        f"Numerical mismatch for M={M} K={K} N={N}: " f"max_diff={(actual - expected).abs().max().item():.4f}"
    )


def test_matmul_sc_validation_rank(device):
    """Verify that rank != 2 raises ValueError."""
    torch_a = torch.randn(1, 32, 32, dtype=torch.bfloat16)
    torch_b = torch.randn(32, 32, dtype=torch.bfloat16)

    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="rank-2"):
        matmul_sc(ttnn_a, ttnn_b)


def test_matmul_sc_validation_inner_dim(device):
    """Verify that mismatched inner dims raises ValueError."""
    torch_a = torch.randn(32, 64, dtype=torch.bfloat16)
    torch_b = torch.randn(32, 32, dtype=torch.bfloat16)  # inner dim mismatch: 64 != 32

    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="inner dimensions"):
        matmul_sc(ttnn_a, ttnn_b)
