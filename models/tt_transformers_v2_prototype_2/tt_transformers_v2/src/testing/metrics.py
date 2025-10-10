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
        elif torch.is_tensor(impl):
            # PyTorch path
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
        elif torch.is_tensor(impl):
            # PyTorch path
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


# Default metrics dictionary for easy import
DEFAULT_METRICS = {
    "max_abs_error": _compute_max_abs_error,
    "mean_abs_error": _compute_mean_abs_error,
    "cosine_similarity": _compute_cosine_similarity,
}


# Public API
__all__ = [
    "_compute_max_abs_error",
    "_compute_mean_abs_error",
    "_compute_cosine_similarity",
    "DEFAULT_METRICS",
]
