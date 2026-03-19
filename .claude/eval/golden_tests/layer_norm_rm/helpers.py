# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for layer_norm_rm golden tests."""

import torch
import ttnn


def pytorch_layer_norm(input_tensor, gamma=None, beta=None, epsilon=1e-5):
    """
    Reference implementation using PyTorch, computed in float32 for precision.
    Returns result in bfloat16 to match expected output dtype.
    """
    x = input_tensor.to(torch.float32)
    W = x.shape[-1]

    if gamma is not None:
        weight = gamma.squeeze().to(torch.float32)
    else:
        weight = None

    if beta is not None:
        bias = beta.squeeze().to(torch.float32)
    else:
        bias = None

    result = torch.nn.functional.layer_norm(x, [W], weight=weight, bias=bias, eps=epsilon)
    return result.to(torch.bfloat16)


def to_ttnn(tensor, device, dtype=ttnn.bfloat16):
    """Helper to convert a torch tensor to a ttnn tensor on device."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def check_output(ttnn_output, expected, shape, rtol, atol):
    """Validate shape, dtype, layout, and numerical correctness."""
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"

    assert ttnn_output.dtype == ttnn.bfloat16, f"Dtype mismatch: got {ttnn_output.dtype}, expected bfloat16"

    assert (
        ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: got {ttnn_output.layout}, expected ROW_MAJOR_LAYOUT"

    actual = ttnn.to_torch(ttnn_output).float()
    expected_f = expected.float()

    abs_diff = (actual - expected_f).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    passing = torch.allclose(actual, expected_f, rtol=rtol, atol=atol)
    assert passing, (
        f"Numerical mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, " f"rtol={rtol}, atol={atol}"
    )
