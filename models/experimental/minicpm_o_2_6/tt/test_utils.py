# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test utilities for MiniCPM-o-2_6 TTNN implementation.

Provides PCC (Pearson Correlation Coefficient) computation and validation helpers.
"""

import torch
import numpy as np
from typing import Union


def compute_pcc(tensor1: Union[torch.Tensor, np.ndarray], tensor2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Pearson Correlation Coefficient (PCC) between two tensors.

    PCC measures the linear correlation between two tensors, ranging from -1 to 1:
    - 1: Perfect positive correlation
    - 0: No correlation
    - -1: Perfect negative correlation

    For model validation, we typically expect PCC >= 0.90 for accurate TTNN implementations.

    Args:
        tensor1: First tensor (PyTorch or NumPy)
        tensor2: Second tensor (PyTorch or NumPy)

    Returns:
        float: Pearson correlation coefficient

    Raises:
        ValueError: If tensors have different shapes
    """
    # Convert to numpy if needed
    if isinstance(tensor1, torch.Tensor):
        # Convert BFloat16 and other dtypes to float32 for numpy compatibility
        if tensor1.dtype != torch.float32:
            tensor1 = tensor1.float()
        tensor1 = tensor1.detach().cpu().numpy()
    if isinstance(tensor2, torch.Tensor):
        # Convert BFloat16 and other dtypes to float32 for numpy compatibility
        if tensor2.dtype != torch.float32:
            tensor2 = tensor2.float()
        tensor2 = tensor2.detach().cpu().numpy()

    # Check shapes match
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensor shapes must match for PCC computation. " f"Got {tensor1.shape} and {tensor2.shape}")

    # Flatten tensors
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()

    # Compute Pearson correlation
    # PCC = cov(X,Y) / (std(X) * std(Y))
    mean1 = np.mean(tensor1_flat)
    mean2 = np.mean(tensor2_flat)

    tensor1_centered = tensor1_flat - mean1
    tensor2_centered = tensor2_flat - mean2

    covariance = np.mean(tensor1_centered * tensor2_centered)
    std1 = np.std(tensor1_flat)
    std2 = np.std(tensor2_flat)

    # Handle edge case where std is zero
    if std1 == 0 or std2 == 0:
        if np.allclose(tensor1_flat, tensor2_flat):
            return 1.0
        else:
            return 0.0

    pcc = covariance / (std1 * std2)

    return float(pcc)


def validate_pcc(
    pcc_value: float,
    threshold: float = 0.90,
    component_name: str = "",
    raise_on_fail: bool = True,
) -> bool:
    """
    Validate that PCC meets the specified threshold.

    Args:
        pcc_value: Computed PCC value
        threshold: Minimum acceptable PCC (default 0.90)
        component_name: Name of component being validated (for error messages)
        raise_on_fail: Whether to raise AssertionError on failure (default True)

    Returns:
        bool: True if PCC meets threshold, False otherwise

    Raises:
        AssertionError: If PCC < threshold and raise_on_fail is True
    """
    passed = pcc_value >= threshold

    if not passed:
        message = f"{component_name} PCC {pcc_value:.6f} < {threshold:.2f}"
        if raise_on_fail:
            raise AssertionError(message)
        else:
            print(f"❌ FAILED: {message}")
            return False
    else:
        print(f"✅ PASSED: {component_name} PCC {pcc_value:.6f} >= {threshold:.2f}")
        return True


def compute_relative_error(
    tensor1: Union[torch.Tensor, np.ndarray],
    tensor2: Union[torch.Tensor, np.ndarray],
    epsilon: float = 1e-8,
) -> float:
    """
    Compute relative error between two tensors.

    Relative error = ||tensor1 - tensor2|| / (||tensor1|| + epsilon)

    Args:
        tensor1: First tensor (reference)
        tensor2: Second tensor (comparison)
        epsilon: Small value to avoid division by zero

    Returns:
        float: Relative error
    """
    # Convert to numpy if needed
    if isinstance(tensor1, torch.Tensor):
        tensor1 = tensor1.detach().cpu().numpy()
    if isinstance(tensor2, torch.Tensor):
        tensor2 = tensor2.detach().cpu().numpy()

    # Compute norms
    diff_norm = np.linalg.norm(tensor1 - tensor2)
    ref_norm = np.linalg.norm(tensor1)

    relative_error = diff_norm / (ref_norm + epsilon)

    return float(relative_error)


def compute_mean_absolute_error(
    tensor1: Union[torch.Tensor, np.ndarray],
    tensor2: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute Mean Absolute Error (MAE) between two tensors.

    MAE = mean(|tensor1 - tensor2|)

    Args:
        tensor1: First tensor
        tensor2: Second tensor

    Returns:
        float: Mean absolute error
    """
    # Convert to numpy if needed
    if isinstance(tensor1, torch.Tensor):
        tensor1 = tensor1.detach().cpu().numpy()
    if isinstance(tensor2, torch.Tensor):
        tensor2 = tensor2.detach().cpu().numpy()

    mae = np.mean(np.abs(tensor1 - tensor2))

    return float(mae)


def print_validation_summary(
    component_name: str,
    pcc: float,
    relative_error: float,
    mae: float,
    threshold: float = 0.90,
):
    """
    Print a formatted validation summary for a component.

    Args:
        component_name: Name of the component
        pcc: Pearson correlation coefficient
        relative_error: Relative error
        mae: Mean absolute error
        threshold: PCC threshold for pass/fail
    """
    passed = pcc >= threshold
    status = "✅ PASSED" if passed else "❌ FAILED"

    print("\n" + "=" * 60)
    print(f"Validation Summary: {component_name}")
    print("=" * 60)
    print(f"Status: {status}")
    print(f"PCC: {pcc:.6f} (threshold: {threshold:.2f})")
    print(f"Relative Error: {relative_error:.6e}")
    print(f"Mean Absolute Error: {mae:.6e}")
    print("=" * 60 + "\n")
