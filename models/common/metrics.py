# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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

from .auto_compose import to_torch_auto_compose

# ======================================================================================
# Public API
# ======================================================================================


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
            abs_diff = _ttnn_op_layout_invariant(diff, ttnn.abs)
            return _ttnn_max_scalar_all_dtype(abs_diff)
        elif _is_ttnn_tensor(impl):
            return (to_torch_auto_compose(impl) - ref).abs().max().item()
        elif _is_ttnn_tensor(ref):
            return (impl - to_torch_auto_compose(ref)).abs().max().item()
        else:
            # PyTorch path - both must be torch tensors
            return (impl - ref).abs().max().item()
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
            abs_diff = _ttnn_op_layout_invariant(diff, ttnn.abs)
            return _ttnn_mean_scalar_all_dtype(abs_diff)
        elif _is_ttnn_tensor(impl):
            return (to_torch_auto_compose(impl) - ref).abs().mean().item()
        elif _is_ttnn_tensor(ref):
            return (impl - to_torch_auto_compose(ref)).abs().mean().item()
        else:
            # PyTorch path - both must be torch tensors
            return (impl - ref).abs().mean().item()
    except Exception as e:
        return float("inf")


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
            return compute_pcc_device(impl, ref)
        elif _is_ttnn_tensor(impl):
            return compute_pcc_host(to_torch_auto_compose(impl), ref)
        elif _is_ttnn_tensor(ref):
            return compute_pcc_host(impl, to_torch_auto_compose(ref))
        else:
            return compute_pcc_host(impl, ref)
    except Exception:
        return 0.0


# Default metrics dictionary for easy import
DEFAULT_METRICS = {
    "max_abs_error": compute_max_abs_error,
    "mean_abs_error": compute_mean_abs_error,
    "pcc": compute_pcc,
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
            diff = _ttnn_op_layout_invariant(ttnn.subtract(impl, ref), ttnn.abs)
            cal_atol = _ttnn_max_scalar_all_dtype(diff)
            # For rtol delta, divide by abs(ref) (may produce inf for zeros; acceptable for reporting)
            divided_by_ref = ttnn.divide(diff, _ttnn_op_layout_invariant(ref, ttnn.abs))
            cal_rtol = _ttnn_max_scalar_all_dtype(divided_by_ref)

            # equal_nan=True semantics and finite/infinite handling
            isnan_impl = _ttnn_op_layout_invariant(impl, ttnn.isnan)
            isnan_ref = _ttnn_op_layout_invariant(ref, ttnn.isnan)
            both_nan = ttnn.logical_and(isnan_impl, isnan_ref)

            isinf_impl = _ttnn_op_layout_invariant(impl, ttnn.isinf)
            isinf_ref = _ttnn_op_layout_invariant(ref, ttnn.isinf)
            impl_sign = _ttnn_op_layout_invariant(impl, ttnn.sign)
            ref_sign = _ttnn_op_layout_invariant(ref, ttnn.sign)
            same_sign_inf = ttnn.eq(impl_sign, ref_sign)
            both_inf_same_sign = ttnn.logical_and(ttnn.logical_and(isinf_impl, isinf_ref), same_sign_inf)

            # Finite elements where numeric closeness applies
            any_nan = ttnn.logical_or(isnan_impl, isnan_ref)
            any_inf = ttnn.logical_or(isinf_impl, isinf_ref)
            finite_both = _ttnn_op_layout_invariant(ttnn.logical_or(any_nan, any_inf), ttnn.logical_not)

            # |impl - ref| <= atol + rtol * |ref|
            bound = ttnn.add(ttnn.mul(_ttnn_op_layout_invariant(ref, ttnn.abs), rtol, dtype=ttnn.bfloat16), atol)
            close_numeric = ttnn.le(diff, bound)
            finite_and_close = ttnn.logical_and(finite_both, close_numeric)

            ok_mask = ttnn.logical_or(ttnn.logical_or(both_nan, both_inf_same_sign), finite_and_close)
            fail_mask = _ttnn_op_layout_invariant(ok_mask, ttnn.logical_not)

            # Reduce to scalar: any failure -> 1.0 else 0.0
            fail_indicator = _ttnn_op_layout_invariant(fail_mask, ttnn.where, true_value=1.0, false_value=0.0)
            any_fail = _ttnn_max_scalar_all_dtype(fail_indicator)
            passing = any_fail == 0.0

            output_str = f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}"
            if not passing:
                output_str += ", Allclose check failed"
            return passing, output_str

        # Fallback: compute with PyTorch (handles mixed inputs by converting TTNN -> torch)
        impl_torch = to_torch_auto_compose(impl) if _is_ttnn_tensor(impl) else impl
        ref_torch = to_torch_auto_compose(ref) if _is_ttnn_tensor(ref) else ref

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
    except Exception as e:
        return False, f"Error computing comp_allclose: {e}"


def compute_pcc_device(impl, ref):
    """Fast on-device PCC for TTNN tensors only."""
    try:
        # Early edge-case handling to mirror CPU semantics
        # - All NaNs → 1.0; mixed NaNs → 0.0
        # - One tensor all zero and the other not → 0.0
        # - Both constant → 1.0 if equal, else 0.0

        # Any nonzero check (all-zero detection)
        impl_abs_max = _ttnn_max_scalar_all_dtype(_ttnn_op_layout_invariant(impl, ttnn.abs))
        ref_abs_max = _ttnn_max_scalar_all_dtype(_ttnn_op_layout_invariant(ref, ttnn.abs))

        impl_has_any = impl_abs_max != 0.0
        ref_has_any = ref_abs_max != 0.0
        if impl_has_any != ref_has_any:
            return 0.0

        # Min/Max scalars for constant and NaN detection
        impl_min = _ttnn_min_scalar_all_dtype(impl)
        impl_max = _ttnn_max_scalar_all_dtype(impl)
        ref_min = _ttnn_min_scalar_all_dtype(ref)
        ref_max = _ttnn_max_scalar_all_dtype(ref)

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
        mean_impl = _ttnn_mean_scalar_all_dtype(impl)
        mean_ref = _ttnn_mean_scalar_all_dtype(ref)

        impl_centered = ttnn.subtract(impl, mean_impl)
        ref_centered = ttnn.subtract(ref, mean_ref)

        # todo)) ttnn.sum() does local reduction only; need CCL reduction for global sum when adding support for multiple-devices
        # [INFO] we cast to float32 to avoid overflow when impl and ref are in bfloat8_b or bfloat4_b
        numerator = ttnn.sum(ttnn.mul(impl_centered, ref_centered, dtype=ttnn.float32))
        impl_sq_sum = ttnn.sum(ttnn.mul(impl_centered, impl_centered, dtype=ttnn.float32))
        ref_sq_sum = ttnn.sum(ttnn.mul(ref_centered, ref_centered, dtype=ttnn.float32))
        denominator = ttnn.sqrt(ttnn.mul(impl_sq_sum, ref_sq_sum, dtype=ttnn.float32))

        # Safe divide
        denom_scalar = denominator.item()
        if denom_scalar == 0.0 or not np.isfinite(denom_scalar):
            return 0.0

        pcc = numerator.item() / denom_scalar
        if not np.isfinite(pcc):
            return 0.0
        return pcc
    except Exception as e:
        # todo)) maybe return a string for logging: f"Error computing PCC on device: {e}, impl: {impl}, ref: {ref}"
        return 0.0


# code stolen from tests/tt_eager/python_api_testing/sweep_tests/comparison_funcs.py
# and models/common/utility_functions.py
def compute_pcc_host(impl, ref):
    """Robust CPU PCC for PyTorch tensors only."""
    try:
        calculated = impl
        golden = ref
        if golden.dtype != calculated.dtype:
            calculated = calculated.type(golden.dtype)

        # Handle complex tensors
        if golden.is_complex() and calculated.is_complex():
            golden = torch.view_as_real(golden.clone())
            calculated = torch.view_as_real(calculated.clone())

        # Convert to float if needed
        if not (golden.is_floating_point() or calculated.is_floating_point()):
            golden = golden.to(torch.float)
            calculated = calculated.to(torch.float)

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
    except Exception:
        return 0.0


# ======================================================================================
# Private Implementation
# ======================================================================================


def _is_ttnn_tensor(x):
    """Safely detect TTNN tensors even if ttnn.Tensor is not defined in this environment."""
    return isinstance(x, ttnn.Tensor)


def _ttnn_op_layout_invariant(x, op_func, **kwargs):
    """
    Generic helper for TTNN operations that require TILE layout for non-sharded tensors.

    Args:
        x: TTNN tensor
        op_func: TTNN operation function to call (e.g., ttnn.sign, ttnn.abs, ttnn.typecast)
        **kwargs: Additional keyword arguments to pass to op_func (e.g., dtype=ttnn.bfloat16 for typecast)

    Returns:
        Result of op_func applied to x, with layout preserved.
    """
    # [ttnn contract] all TTNN operations require tensors to be in TILE layout when working with non-sharded tensors:
    # ttnn.sign(), ttnn.abs(), ttnn.isinf(), ttnn.isnan(), ttnn.typecast(), ttnn.where(), ttnn.logical_not()
    layout = x.get_layout()
    if layout == ttnn.TILE_LAYOUT or x.is_sharded():  # sharded tensors can use either layout
        return op_func(x, **kwargs)
    else:
        return ttnn.to_layout(op_func(ttnn.to_layout(x, ttnn.TILE_LAYOUT), **kwargs), layout)


def _ttnn_max_scalar_all_dtype(x):
    x_bf16 = _ttnn_op_layout_invariant(x, ttnn.typecast, dtype=ttnn.bfloat16)

    # [ttnn contract] ttnn.max() internally calls a FillPad operation, which only supports the following dtypes:
    # BFLOAT16
    # FLOAT32
    # UINT16, UINT32, INT32
    # UINT8
    # see ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_program_factory.hpp for more details
    # [INFO] we cast to bfloat16 to avoid the limitation of the FillPad operation when dealing with e.g., bfloat8_b and bfloat4_b dtypes

    # [ttnn contract] When called without a dim parameter, ttnn.max() returns the maximum value across the entire tensor as a scalar.
    max_val_tensor = ttnn.max(x_bf16)

    # [ttnn contract] When called without a dim parameter, ttnn.max() returns the maximum value across the entire tensor as a scalar.
    # The method supports multiple data types:
    # FLOAT32 → Python float
    # BFLOAT16 → Python float (cast from bfloat16)
    # BFLOAT8_B and BFLOAT4_B → Python float
    return max_val_tensor.item()


def _ttnn_mean_scalar_all_dtype(x):
    x_bf16 = _ttnn_op_layout_invariant(x, ttnn.typecast, dtype=ttnn.bfloat16)

    # [ttnn contract] ttnn.mean() internally calls a FillPad operation, which only supports the following dtypes:
    # BFLOAT16
    # FLOAT32
    # UINT16, UINT32, INT32
    # UINT8
    # see ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_program_factory.hpp for more details
    # [INFO] we cast to bfloat16 to avoid the limitation of the FillPad operation when dealing with e.g., bfloat8_b and bfloat4_b dtypes

    # [ttnn contract] When called without a dim parameter, ttnn.mean() returns the mean value across the entire tensor as a scalar.
    mean_val_tensor = ttnn.mean(x_bf16)
    return mean_val_tensor.item()


def _ttnn_min_scalar_all_dtype(x):
    x_bf16 = _ttnn_op_layout_invariant(x, ttnn.typecast, dtype=ttnn.bfloat16)

    # [ttnn contract] ttnn.mean() internally calls a FillPad operation, which only supports the following dtypes:
    # BFLOAT16
    # FLOAT32
    # UINT16, UINT32, INT32
    # UINT8
    # see ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_program_factory.hpp for more details
    # [INFO] we cast to bfloat16 to avoid the limitation of the FillPad operation when dealing with e.g., bfloat8_b and bfloat4_b dtypes

    # [ttnn contract] When called without a dim parameter, ttnn.mean() returns the mean value across the entire tensor as a scalar.
    min_val_tensor = ttnn.min(x_bf16)
    return min_val_tensor.item()
