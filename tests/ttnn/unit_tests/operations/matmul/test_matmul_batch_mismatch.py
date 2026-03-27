# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for matmul with batch dimension mismatch (in0_B > 1, in1_B = 1).

This tests the reuse_in0_in_CB optimization path that was previously causing hangs
when the input tensor had batch > 1 and the weight tensor had batch = 1.

GitHub Issue: Fixed hang in BEVFormer spatial cross attention with batch=6
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "batch, M, K, N",
    [
        (6, 30125, 256, 256),  # Exact failing case from BEVFormer SCA
        (2, 1000, 256, 256),  # Smaller case with batch=2
        (1, 30125, 256, 256),  # Baseline batch=1 case
    ],
)
def test_matmul_batch_mismatch(batch, M, K, N, device):
    """
    Test ttnn.linear with batch dimension mismatch (input batch >= 1, weight batch = 1).

    This specifically tests the case where:
    - Input: [batch, M, K]
    - Weight: [K, N] (no batch dimension, implicitly batch=1)

    Previously, batch=6 would hang in the matmul kernel due to incorrect handling
    of the reuse_in0_in_CB optimization in the 1D mcast reuse program factory.
    """
    # Create input tensor [batch, M, K]
    input_torch = torch.randn(batch, M, K)
    input_ttnn = ttnn.from_torch(input_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Create weight tensor [K, N]
    weight_torch = torch.randn(K, N)
    weight_ttnn = ttnn.from_torch(weight_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Create bias tensor [N]
    bias_torch = torch.randn(N)
    bias_ttnn = ttnn.from_torch(bias_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Run ttnn.linear (internally uses matmul)
    output = ttnn.linear(input_ttnn, weight_ttnn, bias=bias_ttnn)

    # Ensure operation completes (this was the hang point for batch=6)
    ttnn.synchronize_device(device)

    # Verify output shape
    assert list(output.shape) == [batch, M, N]

    # Convert back to torch for correctness check
    output_torch = ttnn.to_torch(output).to(torch.float32)

    # Compute reference
    expected = torch.matmul(input_torch, weight_torch) + bias_torch

    # Verify correctness (allow for numerical differences due to bfloat16)
    assert torch.allclose(output_torch, expected, rtol=1e-2, atol=1e-1)
