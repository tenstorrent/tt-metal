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

assert hasattr(ttnn, "Tensor")


def _is_ttnn_tensor(x):
    """Safely detect TTNN tensors even if ttnn.Tensor is not defined in this environment."""
    return isinstance(x, ttnn.Tensor)


def compute_max_abs_error(impl, ref):
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
        if _is_ttnn_tensor(impl) and _is_ttnn_tensor(ref):
            # TTNN path - stay on device
            diff = ttnn.subtract(impl, ref)
            abs_diff = ttnn.abs(diff)
            max_val = ttnn.max(abs_diff)
            # todo)) use to_torch_auto_compose here instead of to_torch
            return ttnn.to_torch(max_val).item()  # Convert only final scalar
        elif torch.is_tensor(impl) and torch.is_tensor(ref):
            # PyTorch path - both must be torch tensors
            return (impl - ref).abs().max().item()
        else:
            return float("inf")
    except Exception as e:
        return float("inf")


def compute_mean_abs_error(impl, ref):
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
        if _is_ttnn_tensor(impl) and _is_ttnn_tensor(ref):
            # TTNN path - stay on device
            diff = ttnn.subtract(impl, ref)
            abs_diff = ttnn.abs(diff)
            mean_val = ttnn.mean(abs_diff)
            # todo)) use to_torch_auto_compose here instead of to_torch
            return ttnn.to_torch(mean_val).item()
        elif torch.is_tensor(impl) and torch.is_tensor(ref):
            # PyTorch path - both must be torch tensors
            return (impl - ref).abs().mean().item()
        else:
            return float("inf")
    except Exception as e:
        return float("inf")


# todo)) remove this metric
def compute_cosine_similarity(impl, ref):
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
        if _is_ttnn_tensor(impl) and _is_ttnn_tensor(ref):
            # TTNN path - convert to torch for cosine similarity
            # todo)) we want to avoid this conversion to torch; we want to stay on device; maybe we need to enable different sets of metrics for different device versus host
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
# todo)) maybe it is a good idea to separate device pcc and host pcc? May want to consider this during metrics refactoring on validate_against.py
def compute_pcc(impl, ref):
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
        if _is_ttnn_tensor(impl) and _is_ttnn_tensor(ref):
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
    "max_abs_error": compute_max_abs_error,
    "mean_abs_error": compute_mean_abs_error,
    "pcc": compute_pcc,
    "cosine_similarity": compute_cosine_similarity,
}


# Allclose comparison with detailed delta string
def comp_allclose(impl, ref, rtol=1e-05, atol=1e-08):
    """
    Compare two tensors using an allclose criterion and return (passing, details).

    Provides both a TTNN-native on-device implementation and a PyTorch fallback.
    Mirrors semantics of torch.allclose(..., equal_nan=True) and reports the
    maximum absolute and relative deltas similar to comparison_funcs.py.

    Args:
        impl: Implementation output (PyTorch or TTNN tensor)
        ref: Reference output (PyTorch or TTNN tensor)
        rtol (float): Relative tolerance
        atol (float): Absolute tolerance

    Returns:
        tuple[bool, str]: (passing, "Max ATOL Delta: x, Max RTOL Delta: y[,...]")
    """
    try:
        # TTNN-native path: compute deltas and allclose on device, then transfer final scalars
        if _is_ttnn_tensor(impl) and _is_ttnn_tensor(ref):
            # Compute deltas (device)
            diff = ttnn.abs(ttnn.subtract(impl, ref))
            cal_atol_t = ttnn.max(diff)
            # For rtol delta, divide by abs(ref) (may produce inf for zeros; acceptable for reporting)
            cal_rtol_t = ttnn.max(ttnn.divide(diff, ttnn.abs(ref)))

            # equal_nan=True semantics and finite/infinite handling
            isnan_impl = ttnn.isnan(impl)
            isnan_ref = ttnn.isnan(ref)
            both_nan = ttnn.logical_and(isnan_impl, isnan_ref)

            isinf_impl = ttnn.isinf(impl)
            isinf_ref = ttnn.isinf(ref)
            same_sign_inf = ttnn.eq(ttnn.sign(impl), ttnn.sign(ref))
            both_inf_same_sign = ttnn.logical_and(ttnn.logical_and(isinf_impl, isinf_ref), same_sign_inf)

            # Finite elements where numeric closeness applies
            any_nan = ttnn.logical_or(isnan_impl, isnan_ref)
            any_inf = ttnn.logical_or(isinf_impl, isinf_ref)
            finite_both = ttnn.logical_not(ttnn.logical_or(any_nan, any_inf))

            # |impl - ref| <= atol + rtol * |ref|
            bound = ttnn.add(ttnn.mul(ttnn.abs(ref), rtol), atol)
            close_numeric = ttnn.le(diff, bound)
            finite_and_close = ttnn.logical_and(finite_both, close_numeric)

            ok_mask = ttnn.logical_or(ttnn.logical_or(both_nan, both_inf_same_sign), finite_and_close)
            fail_mask = ttnn.logical_not(ok_mask)

            # Reduce to scalar: any failure -> 1.0 else 0.0
            fail_indicator = ttnn.where(fail_mask, 1.0, 0.0)
            any_fail = ttnn.max(fail_indicator)
            passing = ttnn.to_torch(any_fail).item() == 0.0

            cal_atol = ttnn.to_torch(cal_atol_t).item()
            cal_rtol = ttnn.to_torch(cal_rtol_t).item()
            output_str = f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}"
            if not passing:
                output_str += ", Allclose check failed"
            return passing, output_str

        # Fallback: compute with PyTorch (handles mixed inputs by converting TTNN -> torch)
        impl_torch = ttnn.to_torch(impl) if _is_ttnn_tensor(impl) else impl
        ref_torch = ttnn.to_torch(ref) if _is_ttnn_tensor(ref) else ref

        if torch.is_tensor(impl_torch) and torch.is_tensor(ref_torch):
            # Match dtype for fair comparison
            if impl_torch.dtype != ref_torch.dtype:
                ref_torch = ref_torch.to(impl_torch.dtype)

            atol_delta = torch.max(torch.abs(impl_torch - ref_torch)).item()
            # May produce inf where ref == 0; this mirrors comparison_funcs.py behavior
            rtol_delta = torch.max(torch.abs(impl_torch - ref_torch) / torch.abs(ref_torch)).item()
            passing = torch.allclose(impl_torch, ref_torch, rtol, atol, True)
            output_str = f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}"
            if not passing:
                output_str += ", Allclose check failed"
            return passing, output_str

        # Unsupported types
        return False, "Unsupported input types for comp_allclose"
    except Exception:
        return False, "Error computing comp_allclose"


# Public API
__all__ = [
    "compute_max_abs_error",
    "compute_mean_abs_error",
    "compute_pcc",
    "compute_cosine_similarity",
    "comp_allclose",
    "DEFAULT_METRICS",
]
