# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC (Pearson Correlation Coefficient) validation utilities for RVC TTNN bring-up.

Provides consistent comparison infrastructure between torch reference
and TTNN device outputs.
"""

import torch
from typing import Tuple, Optional


def compute_pcc(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> float:
    """
    Compute Pearson Correlation Coefficient between reference and actual tensors.

    Both tensors are flattened and cast to float32 before comparison.
    Returns a value in [-1.0, 1.0] where 1.0 is perfect correlation.

    Args:
        reference: PyTorch reference output (float32).
        actual: TTNN output converted back to torch (may be bfloat16).

    Returns:
        PCC value as a float.
    """
    ref_flat = reference.detach().float().flatten()
    act_flat = actual.detach().float().flatten()

    # Handle edge cases
    if ref_flat.numel() == 0 or act_flat.numel() == 0:
        return 0.0

    if ref_flat.numel() != act_flat.numel():
        min_len = min(ref_flat.numel(), act_flat.numel())
        ref_flat = ref_flat[:min_len]
        act_flat = act_flat[:min_len]

    # Check for constant tensors (zero variance)
    if ref_flat.std() == 0 and act_flat.std() == 0:
        return 1.0
    if ref_flat.std() == 0 or act_flat.std() == 0:
        return 0.0

    corr_matrix = torch.corrcoef(torch.stack([ref_flat, act_flat]))
    pcc = corr_matrix[0, 1].item()

    # Handle NaN from corrcoef (can happen with near-constant tensors)
    if pcc != pcc:  # NaN check
        return 0.0

    return pcc


def assert_pcc(
    reference: torch.Tensor,
    actual: torch.Tensor,
    threshold: float = 0.999,
    op_name: str = "unknown",
) -> Tuple[bool, float]:
    """
    Assert that PCC between reference and actual exceeds threshold.

    Args:
        reference: PyTorch reference output.
        actual: TTNN output converted back to torch.
        threshold: Minimum acceptable PCC value.
        op_name: Name of the operation for error messages.

    Returns:
        Tuple of (passed, pcc_value).

    Raises:
        AssertionError: If PCC is below threshold.
    """
    pcc = compute_pcc(reference, actual)
    passed = pcc >= threshold

    if not passed:
        raise AssertionError(
            f"PCC validation FAILED for '{op_name}': "
            f"PCC={pcc:.6f} < threshold={threshold:.6f}\n"
            f"  reference shape: {reference.shape}, dtype: {reference.dtype}\n"
            f"  actual shape:    {actual.shape}, dtype: {actual.dtype}\n"
            f"  reference range: [{reference.min():.6f}, {reference.max():.6f}]\n"
            f"  actual range:    [{actual.min():.6f}, {actual.max():.6f}]"
        )

    return passed, pcc


def compute_allclose(
    reference: torch.Tensor,
    actual: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Tuple[bool, float]:
    """
    Compute allclose and max absolute difference.

    Useful as a secondary check alongside PCC.

    Returns:
        Tuple of (allclose_passed, max_abs_diff).
    """
    ref = reference.detach().float()
    act = actual.detach().float()
    max_diff = (ref - act).abs().max().item()
    close = torch.allclose(ref, act, rtol=rtol, atol=atol)
    return close, max_diff


def unpad_from_tile(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Remove TILE padding from a tensor.

    TTNN TILE_LAYOUT pads dimensions to multiples of 32. This function
    slices the tensor back to the original target shape.

    Args:
        tensor: Padded tensor from TTNN.
        target_shape: Original unpadded shape.

    Returns:
        Sliced tensor matching target_shape.
    """
    slices = tuple(slice(0, s) for s in target_shape)
    return tensor[slices]
