# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC (Pearson Correlation Coefficient) tests for TTNN PI0 modules.

This package contains PCC tests that validate TTNN implementations
against PyTorch reference implementations.

Test files:
    - pcc_common.py: Tests for ttnn_common.py utilities
    - pcc_attention.py: Tests for ttnn_attention.py mask utilities
    - pcc_suffix.py: Tests for ttnn_suffix.py embedding
    - pcc_prefix.py: Tests for ttnn_prefix.py embedding
    - pcc_gemma.py: Tests for ttnn_gemma.py transformer blocks
    - pcc_siglip.py: Tests for ttnn_siglip.py vision tower
    - pcc_paligemma.py: Tests for ttnn_paligemma.py backbone
    - pcc_denoise.py: Tests for ttnn_denoise.py denoising
    - pcc_pi0.py: Tests for ttnn_pi0.py full model

PCC Thresholds:
    - 0.99: High precision operations (norms, embeddings)
    - 0.97: Linear operations with bfloat8_b weights
    - 0.95: Complex operations (attention, full blocks)
    - 0.90: End-to-end model outputs
"""

import numpy as np
import torch
from typing import Union

# Check if TTNN is available
try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


def compute_pcc(
    tensor1: Union[torch.Tensor, np.ndarray],
    tensor2: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute Pearson Correlation Coefficient between two tensors.
    
    Args:
        tensor1: First tensor (reference)
        tensor2: Second tensor (comparison)
    
    Returns:
        PCC value between -1 and 1 (1 = perfect correlation)
    """
    if isinstance(tensor1, torch.Tensor):
        t1 = tensor1.detach().float().cpu().numpy().flatten()
    else:
        t1 = tensor1.flatten()
    
    if isinstance(tensor2, torch.Tensor):
        t2 = tensor2.detach().float().cpu().numpy().flatten()
    else:
        t2 = tensor2.flatten()
    
    if len(t1) != len(t2):
        raise ValueError(f"Tensor sizes don't match: {len(t1)} vs {len(t2)}")
    
    # Handle zero variance cases
    std1, std2 = np.std(t1), np.std(t2)
    if std1 == 0 or std2 == 0:
        if np.allclose(t1, t2):
            return 1.0
        else:
            return 0.0
    
    return float(np.corrcoef(t1, t2)[0, 1])


def check_pcc(
    reference: torch.Tensor,
    comparison: torch.Tensor,
    threshold: float = 0.99,
    test_name: str = "unnamed",
    verbose: bool = True,
) -> bool:
    """
    Check if PCC meets threshold and optionally print result.
    
    Args:
        reference: Reference tensor (PyTorch)
        comparison: Comparison tensor (TTNN converted to torch)
        threshold: Minimum acceptable PCC
        test_name: Name of the test for logging
        verbose: Whether to print results
    
    Returns:
        True if PCC >= threshold (with small tolerance for floating point)
    """
    pcc = compute_pcc(reference, comparison)
    # Use small tolerance for floating point comparison
    tolerance = 1e-9
    passed = pcc >= threshold - tolerance
    
    if verbose:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"[{status}] {test_name}: PCC = {pcc:.6f} (threshold: {threshold})")
    
    return passed


def compute_max_abs_error(
    reference: torch.Tensor,
    comparison: torch.Tensor,
) -> float:
    """
    Compute maximum absolute error between tensors.
    
    Args:
        reference: Reference tensor
        comparison: Comparison tensor
    
    Returns:
        Maximum absolute difference
    """
    if isinstance(reference, torch.Tensor):
        ref = reference.detach().float().cpu()
    else:
        ref = torch.tensor(reference, dtype=torch.float32)
    
    if isinstance(comparison, torch.Tensor):
        comp = comparison.detach().float().cpu()
    else:
        comp = torch.tensor(comparison, dtype=torch.float32)
    
    return float((ref - comp).abs().max())


def compute_mean_abs_error(
    reference: torch.Tensor,
    comparison: torch.Tensor,
) -> float:
    """
    Compute mean absolute error between tensors.
    
    Args:
        reference: Reference tensor
        comparison: Comparison tensor
    
    Returns:
        Mean absolute difference
    """
    if isinstance(reference, torch.Tensor):
        ref = reference.detach().float().cpu()
    else:
        ref = torch.tensor(reference, dtype=torch.float32)
    
    if isinstance(comparison, torch.Tensor):
        comp = comparison.detach().float().cpu()
    else:
        comp = torch.tensor(comparison, dtype=torch.float32)
    
    return float((ref - comp).abs().mean())


def ttnn_to_torch(tensor: "ttnn.Tensor") -> torch.Tensor:
    """Convert TTNN tensor to PyTorch tensor."""
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")
    return ttnn.to_torch(tensor)


def torch_to_ttnn(
    tensor: torch.Tensor,
    device: "ttnn.Device",
    dtype: "ttnn.DataType" = None,
    layout: "ttnn.Layout" = None,
) -> "ttnn.Tensor":
    """Convert PyTorch tensor to TTNN tensor."""
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")
    
    if dtype is None:
        dtype = ttnn.bfloat16
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

