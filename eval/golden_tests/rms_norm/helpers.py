# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for rms_norm golden tests."""

import torch
import ttnn


def pytorch_rms_norm(input_tensor, *, gamma=None, epsilon=1e-6):
    """
    Reference RMSNorm implementation using PyTorch, computed in float32 for precision.

    RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

    Returns result in the same dtype as the input tensor.
    """
    original_dtype = input_tensor.dtype
    x = input_tensor.to(torch.float32)
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + epsilon)
    result = x / rms
    if gamma is not None:
        result = result * gamma.to(torch.float32)
    return result.to(original_dtype)


def to_ttnn(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Helper to convert a torch tensor to a ttnn tensor on device."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def check_output(
    ttnn_output, expected, shape, rtol, atol, expected_dtype=ttnn.bfloat16, expected_layout=ttnn.TILE_LAYOUT
):
    """Validate shape, dtype, layout, and numerical correctness."""
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"

    assert ttnn_output.dtype == expected_dtype, f"Dtype mismatch: got {ttnn_output.dtype}, expected {expected_dtype}"

    assert (
        ttnn_output.layout == expected_layout
    ), f"Layout mismatch: got {ttnn_output.layout}, expected {expected_layout}"

    actual = ttnn.to_torch(ttnn_output).float()
    expected_f = expected.float()

    abs_diff = (actual - expected_f).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    passing = torch.allclose(actual, expected_f, rtol=rtol, atol=atol)
    assert passing, (
        f"Numerical mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, " f"rtol={rtol}, atol={atol}"
    )
