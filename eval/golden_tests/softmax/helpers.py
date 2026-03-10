# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for softmax golden tests."""

import torch
import ttnn


def pytorch_softmax(input_tensor, dim=-1, *, numeric_stable=True):
    """
    Reference implementation using PyTorch, computed in float32 for precision.
    Returns result in bfloat16 to match expected output dtype.

    The numeric_stable parameter mirrors the TTNN op signature but does not
    change the reference computation — torch.softmax is always numerically
    stable internally. We keep the parameter so golden tests can call the
    reference with the same arguments they pass to the TTNN op.
    """
    x = input_tensor.to(torch.float32)
    result = torch.softmax(x, dim=dim)
    return result.to(torch.bfloat16)


def to_ttnn(tensor, device, dtype=ttnn.bfloat16):
    """Helper to convert a torch tensor to a ttnn tensor on device."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def check_output(ttnn_output, expected, shape, rtol, atol):
    """Validate shape, dtype, layout, and numerical correctness."""
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"

    assert ttnn_output.dtype == ttnn.bfloat16, f"Dtype mismatch: got {ttnn_output.dtype}, expected bfloat16"

    assert ttnn_output.layout == ttnn.TILE_LAYOUT, f"Layout mismatch: got {ttnn_output.layout}, expected TILE_LAYOUT"

    actual = ttnn.to_torch(ttnn_output).float()
    expected_f = expected.float()

    abs_diff = (actual - expected_f).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    passing = torch.allclose(actual, expected_f, rtol=rtol, atol=atol)
    assert passing, (
        f"Numerical mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, " f"rtol={rtol}, atol={atol}"
    )
