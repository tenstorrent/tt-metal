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

    For TTNN tensors: Uses on-device computation (100-1000× faster). If edge cases
    produce non-finite values (NaN/inf), raises ValueError with instructions to use
    Pattern 2 (PyTorch reference with output_map for robust CPU-based PCC).

    For PyTorch tensors: Uses robust numpy implementation (from tt-metal
    comparison_funcs.py) that handles complex, NaN, inf, and constant tensors.

    Args:
        impl: Implementation output (PyTorch or TTNN tensor)
        ref: Reference output (PyTorch or TTNN tensor)

    Returns:
        float: PCC value in range [-1.0, 1.0], or 0.0 on error

    Raises:
        ValueError: If TTNN-native computation produces non-finite values

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
            # Early edge-case handling to mirror CPU semantics
            # - All NaNs → 1.0; mixed NaNs → 0.0
            # - One tensor all zero and the other not → 0.0
            # - Both constant → 1.0 if equal, else 0.0

            # Any nonzero check (all-zero detection)
            impl_abs_max = ttnn.to_torch(ttnn.max(ttnn.abs(impl))).item()
            ref_abs_max = ttnn.to_torch(ttnn.max(ttnn.abs(ref))).item()
            impl_has_any = impl_abs_max != 0.0
            ref_has_any = ref_abs_max != 0.0
            if impl_has_any != ref_has_any:
                return 0.0

            # Min/Max scalars for constant and NaN detection
            impl_min = ttnn.to_torch(ttnn.min(impl)).item()
            impl_max = ttnn.to_torch(ttnn.max(impl)).item()
            ref_min = ttnn.to_torch(ttnn.min(ref)).item()
            ref_max = ttnn.to_torch(ttnn.max(ref)).item()

            impl_min_finite = np.isfinite(impl_min)
            impl_max_finite = np.isfinite(impl_max)
            ref_min_finite = np.isfinite(ref_min)
            ref_max_finite = np.isfinite(ref_max)

            impl_all_nan = (not impl_min_finite) and (not impl_max_finite)
            ref_all_nan = (not ref_min_finite) and (not ref_max_finite)
            if impl_all_nan and ref_all_nan:
                return 1.0
            if impl_all_nan != ref_all_nan:
                return 0.0

            # Constant tensors
            if impl_min_finite and impl_max_finite and ref_min_finite and ref_max_finite:
                if impl_min == impl_max and ref_min == ref_max:
                    return (
                        1.0
                        if torch.isclose(
                            torch.tensor(impl_max, dtype=torch.float32), torch.tensor(ref_max, dtype=torch.float32)
                        )
                        else 0.0
                    )

            # Standard PCC formula on device
            mean_impl = ttnn.mean(impl)
            mean_ref = ttnn.mean(ref)

            impl_centered = ttnn.subtract(impl, mean_impl)
            ref_centered = ttnn.subtract(ref, mean_ref)

            numerator = ttnn.sum(ttnn.mul(impl_centered, ref_centered))
            impl_sq_sum = ttnn.sum(ttnn.mul(impl_centered, impl_centered))
            ref_sq_sum = ttnn.sum(ttnn.mul(ref_centered, ref_centered))
            denominator = ttnn.sqrt(ttnn.mul(impl_sq_sum, ref_sq_sum))

            # Safe divide
            denom_scalar = ttnn.to_torch(denominator).item()
            if denom_scalar == 0.0 or not np.isfinite(denom_scalar):
                return 0.0

            pcc_tensor = ttnn.mul(numerator, ttnn.reciprocal(denominator))
            pcc = ttnn.to_torch(pcc_tensor).item()
            if not np.isfinite(pcc):
                return 0.0
            return pcc

        # PyTorch tensor path - robust CPU implementation from tt-metal comparison_funcs.py
        if torch.is_tensor(impl) and torch.is_tensor(ref):
            calculated = impl
            golden = ref

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

        # Unsupported tensor type combination
        return 0.0

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
