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
    ttnn_output,
    expected,
    shape,
    pcc_threshold=0.999,
    rms_threshold=0.02,
    expected_dtype=ttnn.bfloat16,
    expected_layout=ttnn.TILE_LAYOUT,
):
    """Validate shape, dtype, layout, and numerical correctness using PCC + RMS."""
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"

    assert ttnn_output.dtype == expected_dtype, f"Dtype mismatch: got {ttnn_output.dtype}, expected {expected_dtype}"

    assert (
        ttnn_output.layout == expected_layout
    ), f"Layout mismatch: got {ttnn_output.layout}, expected {expected_layout}"

    actual = ttnn.to_torch(ttnn_output).to(torch.float64)
    expected_f = expected.to(torch.float64)

    # PCC (Pearson Correlation Coefficient)
    a_flat = actual.flatten()
    e_flat = expected_f.flatten()
    a_centered = a_flat - a_flat.mean()
    e_centered = e_flat - e_flat.mean()
    num = (a_centered * e_centered).sum()
    den = a_centered.norm() * e_centered.norm()
    pcc = (num / den).item() if den > 1e-30 else (1.0 if num.abs() < 1e-30 else 0.0)

    # RMS error
    rms_err = ((actual - expected_f) ** 2).mean().sqrt().item()

    assert pcc >= pcc_threshold, f"PCC too low: {pcc:.8f} < {pcc_threshold} (rms_error={rms_err:.6f})"
    assert rms_err <= rms_threshold, f"RMS error too high: {rms_err:.6f} > {rms_threshold} (pcc={pcc:.8f})"
