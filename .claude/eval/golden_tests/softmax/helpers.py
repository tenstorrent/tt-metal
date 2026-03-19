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


def check_output(ttnn_output, expected, shape, pcc_threshold=0.999, rms_threshold=0.02):
    """Validate shape, dtype, layout, and numerical correctness using PCC + RMS."""
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"

    assert ttnn_output.dtype == ttnn.bfloat16, f"Dtype mismatch: got {ttnn_output.dtype}, expected bfloat16"

    assert ttnn_output.layout == ttnn.TILE_LAYOUT, f"Layout mismatch: got {ttnn_output.layout}, expected TILE_LAYOUT"

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
