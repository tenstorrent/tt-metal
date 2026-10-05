# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for matmul with batch dimension mismatch.

test_matmul_a_batch1_b_batched: A batch=1, B batch>1 (issue #12834)
test_matmul_batch_mismatch:     A batch>1, B batch=1 (prior regression)
test_matmul_batch_broadcast:    torch-style broadcast of mixed / rank-mismatched batch dims
"""

import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((1, 1, 128, 2048), (1, 7, 2048, 256)),
        ((1, 128, 2048), (7, 2048, 256)),
        ((1, 1, 256, 512), (1, 3, 512, 64)),
        ((1, 64, 768), (5, 768, 192)),
        ((1, 1, 1, 128, 512), (2, 3, 4, 512, 64)),
        ((1, 1, 1, 1, 64, 512), (2, 3, 4, 5, 512, 128)),
    ],
)
def test_matmul_a_batch1_b_batched(a_shape, b_shape, device):
    """
    Test ttnn.matmul where A has batch=1 and B has batch>1 (issue #12834).
    Previously crashed with "bmm expects input tensors of shapes BCMK*BCKN=BCMN".
    The auto-selector should now pick 1D mcast config and use in0_reuse.
    """
    torch.manual_seed(0)
    a_torch = torch.randn(*a_shape)
    b_torch = torch.randn(*b_shape)

    a_ttnn = ttnn.from_torch(a_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(b_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.matmul(a_ttnn, b_ttnn)
    ttnn.synchronize_device(device)

    expected_shape = list(torch.matmul(a_torch, b_torch).shape)
    assert list(output.shape) == expected_shape

    output_torch = ttnn.to_torch(output).to(torch.float32)
    expected = torch.matmul(a_torch, b_torch)

    pcc_passed, pcc_message = comp_pcc(expected, output_torch, 0.99)
    assert pcc_passed, f"PCC check failed: {pcc_message}"


@pytest.mark.parametrize(
    "batch, M, K, N",
    [
        (6, 3000, 256, 256),
    ],
)
def test_matmul_batch_mismatch(batch, M, K, N, device):
    """
    Test ttnn.linear with batch dimension mismatch (input batch >= 1, weight batch = 1).

    This specifically tests the case where:
    - Input: [batch, M, K]
    - Weight: [K, N] (no batch dimension, implicitly batch=1)

    Previously, batch=6 would hang in the matmul kernel due to incorrect argument passed to the in1 reader receiver kernel.
    """
    torch.manual_seed(0)
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

    # Verify correctness using PCC (allow for numerical differences due to bfloat16)
    pcc_passed, pcc_message = comp_pcc(expected, output_torch, 0.99)
    assert pcc_passed, f"PCC check failed: {pcc_message}"


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((1, 3, 64, 96), (2, 1, 96, 128)),
        ((2, 1, 64, 96), (1, 3, 96, 128)),
        ((64, 96), (2, 3, 96, 128)),
        ((3, 64, 96), (2, 3, 96, 128)),
        ((2, 3, 64, 96), (3, 96, 128)),
        ((2, 3, 4, 64, 96), (1, 3, 1, 96, 128)),
        ((2, 3, 37, 50), (1, 3, 50, 70)),
        ((2, 4, 64, 96), (2, 1, 96, 128)),
        ((3, 1, 2, 1, 4, 1, 64, 96), (4, 2, 96, 128)),
    ],
)
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("core_grid", [None, ttnn.CoreGrid(y=8, x=8)])
def test_matmul_batch_broadcast(a_shape, b_shape, transpose_a, transpose_b, core_grid, device):
    torch.manual_seed(0)
    a_torch = torch.randn(*a_shape)
    b_torch = torch.randn(*b_shape)
    a_dev = a_torch.transpose(-1, -2) if transpose_a else a_torch
    b_dev = b_torch.transpose(-1, -2) if transpose_b else b_torch
    a_ttnn = ttnn.from_torch(a_dev, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(b_dev, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.matmul(a_ttnn, b_ttnn, transpose_a=transpose_a, transpose_b=transpose_b, core_grid=core_grid)

    expected = torch.matmul(a_torch, b_torch)
    assert list(output.shape) == list(expected.shape)
    pcc_passed, pcc_message = comp_pcc(expected, ttnn.to_torch(output).to(torch.float32), 0.999)
    assert pcc_passed, f"PCC check failed: {pcc_message}"


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape",
    [
        ((2, 3, 64, 96), (128, 96), (1, 1)),
        ((2, 3, 64, 96), (128, 96), (1,)),
        ((2, 3, 64, 96), (128, 96), (1, 128)),
        ((2, 3, 64, 96), (2, 1, 128, 96), (1, 128)),
        ((2, 3, 64, 96), (2, 1, 128, 96), (1, 1)),
        ((5, 37, 50), (70, 50), (1, 1)),
        ((2, 1, 2, 3, 2, 2, 64, 96), (128, 96), (1, 1)),
    ],
)
def test_linear_broadcast_weight_and_bias(input_shape, weight_shape, bias_shape, device):
    torch.manual_seed(0)
    input_torch = torch.randn(*input_shape)
    weight_torch = torch.randn(*weight_shape)
    bias_torch = torch.randn(*bias_shape)
    to_dev = lambda t: ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.linear(to_dev(input_torch), to_dev(weight_torch), bias=to_dev(bias_torch), transpose_b=True)

    expected = torch.matmul(input_torch, weight_torch.transpose(-1, -2)) + bias_torch
    assert list(output.shape) == list(expected.shape)
    pcc_passed, pcc_message = comp_pcc(expected, ttnn.to_torch(output).to(torch.float32), 0.999)
    assert pcc_passed, f"PCC check failed: {pcc_message}"


def test_matmul_batch_not_broadcastable(device, expect_error):
    a_ttnn = ttnn.from_torch(torch.randn(2, 64, 96), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.from_torch(torch.randn(3, 96, 128), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "are not broadcastable"):
        ttnn.matmul(a_ttnn, b_ttnn)
