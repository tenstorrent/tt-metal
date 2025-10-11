#!/usr/bin/env python3
"""
TTNN Metric Functions

Metric functions for comparing TTNN implementations against reference implementations.
All metrics support both PyTorch and TTNN tensors, with TTNN-native computation
that stays on device until the final scalar result.

Key Features:
- Automatic detection of tensor type (PyTorch vs TTNN)
- TTNN-native computation using device operations
- Minimal host transfers (only final scalar)
- Graceful fallback to PyTorch for non-tensor inputs
"""

import numpy as np
import torch
import ttnn


def _compute_max_abs_error(impl, ref):
    """
    Compute maximum absolute error between two tensors.

    Supports both PyTorch and TTNN tensors. For TTNN tensors, computation
    stays on device until the final scalar value.

    Args:
        impl: Implementation output (PyTorch or TTNN tensor)
        ref: Reference output (PyTorch or TTNN tensor)

    Returns:
        float: Maximum absolute difference between tensors

    Examples:
        >>> a = torch.tensor([1.0, 2.0, 3.0])
        >>> b = torch.tensor([1.1, 2.0, 2.9])
        >>> _compute_max_abs_error(a, b)
        0.10000002384185791
    """
    try:
        if isinstance(impl, ttnn.Tensor) and isinstance(ref, ttnn.Tensor):
            # TTNN path - stay on device
            diff = ttnn.subtract(impl, ref)
            abs_diff = ttnn.abs(diff)
            max_val = ttnn.max(abs_diff)
            return ttnn.to_torch(max_val).item()  # Convert only final scalar
        elif torch.is_tensor(impl) and torch.is_tensor(ref):
            # PyTorch path - both must be torch tensors
            return (impl - ref).abs().max().item()
        else:
            return float("inf")
    except Exception as e:
        return float("inf")


def _compute_mean_abs_error(impl, ref):
    """
    Compute mean absolute error between two tensors.

    Supports both PyTorch and TTNN tensors. For TTNN tensors, computation
    stays on device until the final scalar value.

    Args:
        impl: Implementation output (PyTorch or TTNN tensor)
        ref: Reference output (PyTorch or TTNN tensor)

    Returns:
        float: Mean absolute difference between tensors

    Examples:
        >>> a = torch.tensor([1.0, 2.0, 3.0])
        >>> b = torch.tensor([1.1, 2.0, 2.9])
        >>> _compute_mean_abs_error(a, b)
        0.06666667014360428
    """
    try:
        if isinstance(impl, ttnn.Tensor) and isinstance(ref, ttnn.Tensor):
            # TTNN path - stay on device
            diff = ttnn.subtract(impl, ref)
            abs_diff = ttnn.abs(diff)
            mean_val = ttnn.mean(abs_diff)
            return ttnn.to_torch(mean_val).item()
        elif torch.is_tensor(impl) and torch.is_tensor(ref):
            # PyTorch path - both must be torch tensors
            return (impl - ref).abs().mean().item()
        else:
            return float("inf")
    except Exception as e:
        return float("inf")


def _compute_cosine_similarity(impl, ref):
    """
    Compute cosine similarity between two tensors.

    Measures the cosine of the angle between two vectors. Returns 1.0 for
    identical directions, 0.0 for orthogonal, and -1.0 for opposite directions.

    Note: For TTNN tensors, converts to PyTorch before computing cosine similarity
    as TTNN doesn't have a built-in cosine similarity operation.

    Args:
        impl: Implementation output (PyTorch or TTNN tensor)
        ref: Reference output (PyTorch or TTNN tensor)

    Returns:
        float: Cosine similarity in range [-1.0, 1.0]

    Examples:
        >>> a = torch.tensor([1.0, 0.0])
        >>> b = torch.tensor([1.0, 0.0])
        >>> _compute_cosine_similarity(a, b)
        1.0

        >>> a = torch.tensor([1.0, 0.0])
        >>> b = torch.tensor([0.0, 1.0])
        >>> _compute_cosine_similarity(a, b)
        0.0
    """
    try:
        if isinstance(impl, ttnn.Tensor) and isinstance(ref, ttnn.Tensor):
            # TTNN path - convert to torch for cosine similarity
            impl_torch = ttnn.to_torch(impl).flatten()
            ref_torch = ttnn.to_torch(ref).flatten()
            return torch.nn.functional.cosine_similarity(impl_torch, ref_torch, dim=0).item()
        elif torch.is_tensor(impl) and torch.is_tensor(ref):
            # PyTorch path
            return torch.nn.functional.cosine_similarity(impl.flatten(), ref.flatten(), dim=0).item()
        else:
            return 0.0
    except Exception as e:
        return 0.0


# code stolen from tests/tt_eager/python_api_testing/sweep_tests/comparison_funcs.py
def _compute_pcc(impl, ref):
    """
    Compute Pearson Correlation Coefficient (PCC) between two tensors.

    PCC measures the linear correlation between two tensors. Returns 1.0 for
    perfect positive correlation, 0.0 for no correlation, and -1.0 for perfect
    negative correlation. This is a common metric used in tt-metal for validating
    tensor computations.

    For TTNN tensors, uses on-device computation when possible. Falls back to
    robust numpy implementation (from tt-metal comparison_funcs.py) for edge cases.

    Args:
        impl: Implementation output (PyTorch or TTNN tensor)
        ref: Reference output (PyTorch or TTNN tensor)

    Returns:
        float: PCC value in range [-1.0, 1.0], or 0.0 on error

    Examples:
        >>> a = torch.tensor([1.0, 2.0, 3.0])
        >>> b = torch.tensor([1.0, 2.0, 3.0])
        >>> _compute_pcc(a, b)
        1.0

        >>> a = torch.tensor([1.0, 2.0, 3.0])
        >>> b = torch.tensor([3.0, 2.0, 1.0])
        >>> _compute_pcc(a, b)
        -1.0
    """
    try:
        # TTNN fast path - compute on device
        if isinstance(impl, ttnn.Tensor) and isinstance(ref, ttnn.Tensor):
            # PCC formula: sum((x - mean_x) * (y - mean_y)) / sqrt(sum((x - mean_x)^2) * sum((y - mean_y)^2))
            mean_impl = ttnn.mean(impl)
            mean_ref = ttnn.mean(ref)

            # Center the tensors
            impl_centered = ttnn.subtract(impl, mean_impl)
            ref_centered = ttnn.subtract(ref, mean_ref)

            # Numerator: sum of element-wise product
            numerator = ttnn.sum(ttnn.mul(impl_centered, ref_centered))

            # Denominator: sqrt(sum(x^2) * sum(y^2))
            impl_sq_sum = ttnn.sum(ttnn.mul(impl_centered, impl_centered))
            ref_sq_sum = ttnn.sum(ttnn.mul(ref_centered, ref_centered))
            denominator = ttnn.sqrt(ttnn.mul(impl_sq_sum, ref_sq_sum))

            # Final PCC - only transfer scalar to host
            pcc_tensor = ttnn.mul(numerator, ttnn.reciprocal(denominator))
            pcc = ttnn.to_torch(pcc_tensor).item()

            # Check for NaN/inf (edge cases) - if found, fall back to robust version
            if not (np.isfinite(pcc)):
                # Fall through to robust CPU implementation below
                pass
            else:
                return pcc

        # Convert to torch tensors for robust CPU implementation
        if isinstance(impl, ttnn.Tensor):
            calculated = ttnn.to_torch(impl)
        elif torch.is_tensor(impl):
            calculated = impl
        else:
            return 0.0

        if isinstance(ref, ttnn.Tensor):
            golden = ttnn.to_torch(ref)
        elif torch.is_tensor(ref):
            golden = ref
        else:
            return 0.0

        # Handle complex tensors
        if golden.is_complex() and calculated.is_complex():
            golden = torch.view_as_real(golden.clone())
            calculated = torch.view_as_real(calculated.clone())

        # Convert to float if needed
        if not (golden.is_floating_point() or calculated.is_floating_point()):
            golden = golden.to(torch.float)
            calculated = calculated.to(torch.float)

        # PCC computation from tt-metal comparison_funcs.py
        # Both tensors are nan
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            return 1.0

        # One tensor is all nan, the other is not
        if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            return 0.0

        # One tensor is all zero, the other is not
        if torch.any(golden.bool()) != torch.any(calculated.bool()):
            return 0.0

        # Mask all infs and nans
        golden = golden.clone()
        golden[
            torch.logical_or(
                torch.isnan(golden),
                torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
            )
        ] = 0
        calculated = calculated.clone()
        calculated[
            torch.logical_or(
                torch.isnan(calculated),
                torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
            )
        ] = 0

        if torch.equal(golden, calculated):
            return 1.0

        if golden.dtype == torch.bfloat16:
            golden = golden.type(torch.float32)
            calculated = calculated.type(torch.float32)

        # Single element case
        if golden.numel() == 1:
            return float(torch.equal(golden, calculated))

        # If both tensors are constant
        if torch.max(golden) == torch.min(golden) and torch.max(calculated) == torch.min(calculated):
            return torch.isclose(torch.max(golden), torch.max(calculated)).item()

        # Compute PCC using numpy's corrcoef
        cal_pcc = np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
        # Remove correlation coefficient with self (typically always 1.0)
        mask = np.ones(cal_pcc.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        cal_pcc = np.min(cal_pcc[mask])

        if isinstance(cal_pcc, np.ma.core.MaskedConstant):
            return 1.0

        return float(cal_pcc)

    except Exception as e:
        return 0.0


# Default metrics dictionary for easy import
DEFAULT_METRICS = {
    "max_abs_error": _compute_max_abs_error,
    "mean_abs_error": _compute_mean_abs_error,
    "pcc": _compute_pcc,
    "cosine_similarity": _compute_cosine_similarity,
}


# Public API
__all__ = [
    "_compute_max_abs_error",
    "_compute_mean_abs_error",
    "_compute_pcc",
    "_compute_cosine_similarity",
    "DEFAULT_METRICS",
]
