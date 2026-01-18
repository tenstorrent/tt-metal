# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def torch_rms_norm(x, gamma=None, eps=1e-5):
    """PyTorch reference implementation of RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if gamma is not None:
        x_normed = x_normed * gamma
    return x_normed


def torch_layer_norm(x, gamma=None, beta=None, residual=None, eps=1e-5):
    """PyTorch reference implementation of LayerNorm."""
    if residual is not None:
        x = x + residual
    mean = x.mean(-1, keepdim=True)
    variance = x.var(-1, keepdim=True, unbiased=False)
    x_normed = (x - mean) / torch.sqrt(variance + eps)
    if gamma is not None:
        x_normed = x_normed * gamma
    if beta is not None:
        x_normed = x_normed + beta
    return x_normed


def assert_with_pcc(expected, actual, pcc=0.99):
    """Assert that actual matches expected within PCC tolerance."""
    expected_flat = expected.flatten().float()
    actual_flat = actual.flatten().float()
    correlation = torch.corrcoef(torch.stack([expected_flat, actual_flat]))[0, 1]
    assert correlation >= pcc, f"PCC {correlation} < {pcc}"


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sequential_two_rms_norm(device, batch_size, h, w):
    """Test sequential execution of two RMS norm operations."""
    torch.manual_seed(42)

    # Create inputs
    torch_input1 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input2 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Compute torch reference
    torch_output1 = torch_rms_norm(torch_input1, eps=1e-5)
    torch_output2 = torch_rms_norm(torch_input2, eps=1e-6)

    # Move inputs to device
    input1 = ttnn.from_torch(torch_input1, device=device, layout=ttnn.TILE_LAYOUT)
    input2 = ttnn.from_torch(torch_input2, device=device, layout=ttnn.TILE_LAYOUT)

    # Execute sequentially
    results = ttnn.sequential(
        [
            (ttnn.rms_norm, input1, {"epsilon": 1e-5}),
            (ttnn.rms_norm, input2, {"epsilon": 1e-6}),
        ]
    )

    # Verify results
    output1 = ttnn.to_torch(ttnn.from_device(results[0]))
    output2 = ttnn.to_torch(ttnn.from_device(results[1]))

    assert_with_pcc(torch_output1, output1, 0.999)
    assert_with_pcc(torch_output2, output2, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sequential_rms_and_layernorm(device, batch_size, h, w):
    """Test sequential execution of RMS norm followed by LayerNorm."""
    torch.manual_seed(42)

    # Create inputs
    torch_input1 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input2 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    # Compute torch reference
    torch_output1 = torch_rms_norm(torch_input1, eps=1e-5)
    torch_output2 = torch_layer_norm(torch_input2, torch_weight, torch_bias, eps=1e-6)

    # Move inputs to device
    input1 = ttnn.from_torch(torch_input1, device=device, layout=ttnn.TILE_LAYOUT)
    input2 = ttnn.from_torch(torch_input2, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    bias = ttnn.from_torch(torch_bias, device=device, layout=ttnn.TILE_LAYOUT)

    # Execute sequentially
    results = ttnn.sequential(
        [
            (ttnn.rms_norm, input1, {"epsilon": 1e-5}),
            (ttnn.layer_norm, input2, {"epsilon": 1e-6, "weight": weight, "bias": bias}),
        ]
    )

    # Verify results
    output1 = ttnn.to_torch(ttnn.from_device(results[0]))
    output2 = ttnn.to_torch(ttnn.from_device(results[1]))

    assert_with_pcc(torch_output1, output1, 0.999)
    assert_with_pcc(torch_output2, output2, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sequential_chained_ops(device, batch_size, h, w):
    """Test sequential execution where output of one op is input to next."""
    torch.manual_seed(42)

    # Create input
    torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Compute torch reference: RMS norm followed by RMS norm again
    torch_intermediate = torch_rms_norm(torch_input, eps=1e-5)
    torch_output = torch_rms_norm(torch_intermediate, eps=1e-6)

    # Move input to device
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Execute first op
    result1 = ttnn.rms_norm(input_tensor, epsilon=1e-5)

    # Execute second op using result of first
    results = ttnn.sequential(
        [
            (ttnn.rms_norm, result1, {"epsilon": 1e-6}),
        ]
    )

    # Verify final result
    output = ttnn.to_torch(ttnn.from_device(results[0]))
    assert_with_pcc(torch_output, output, 0.999)
