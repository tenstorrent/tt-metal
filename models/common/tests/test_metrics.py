# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for TTNN-native metric functions - NO MOCKS.
Tests numerical correctness of metrics computed with actual TTNN ops.

Run with: pytest test_ttnn_metrics_numerical.py -v
Requires: TTNN hardware/installation, pytest
"""

import pytest
import torch

import ttnn

# Import metric functions from the validation framework
from models.common.metrics import comp_allclose, compute_max_abs_error, compute_mean_abs_error, compute_pcc

pytestmark = [
    pytest.mark.parametrize(
        "ttnn_mesh_device",
        [
            (1, 1),
            # todo)) currently device metrics are only supported on 1x1 mesh device, which is experimental feature.
            # we could add support for more mesh shapes later if there is demand.
            # (1, 2),
            # (1, 8),
            # (2, 4),
        ],
        ids=[
            "1x1",
            # "1x2",
            # "1x8",
            # "2x4",
        ],
        indirect=True,
    ),
    pytest.mark.parametrize(
        "layout,dtype",
        [
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
            (ttnn.TILE_LAYOUT, ttnn.bfloat16),
            (ttnn.TILE_LAYOUT, ttnn.bfloat8_b),  # quantized dtypes only supported with TILE_LAYOUT
            (ttnn.TILE_LAYOUT, ttnn.bfloat4_b),  # quantized dtypes only supported with TILE_LAYOUT
        ],
        ids=["row_major_bf16", "tile_bf16", "tile_bf8b", "tile_bf4b"],
    ),
]


def _quantize_like_ttnn(x: torch.Tensor, device: ttnn.MeshDevice, dtype, layout):
    """Round-trip through TTNN to obtain torch tensor quantized like given dtype.

    For bf16, returns input unchanged. For bf8b/bf4b, uses TILE_LAYOUT only.
    """
    x_t = ttnn.from_torch(x, device=device, dtype=dtype, layout=layout)
    return ttnn.to_torch(x_t)


# Test case definitions
# Format: (name, tensor_a_fn, tensor_b_fn, max_spec, mean_spec, cosine_spec)
# Each spec is a tuple: (expected_value, tolerance) or None to skip expected check
TEST_CASES = [
    # Identical tensors
    pytest.param(
        "identical_random",
        lambda: torch.randn(32, 64, dtype=torch.bfloat16),
        lambda t: t.clone(),
        (0.0, 1e-6),  # max: expect ~0, tight tolerance
        (0.0, 1e-6),  # mean: expect ~0, tight tolerance
        id="identical_random",
    ),
    # Known differences
    pytest.param(
        "known_diff_0.5",
        lambda: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16),
        lambda t: t + torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=torch.bfloat16),
        (0.5, 0.02),  # max: expect 0.5
        None,  # mean: skip expected check, just compare TTNN vs PyTorch
        id="known_diff_max_0.5",
    ),
    pytest.param(
        "uniform_diff_0.5",
        lambda: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        lambda t: t + 1,
        (1, 0.02),  # max: expect 0.5
        (1, 0.02),  # mean: expect 0.5 (all elements differ by same amount)
        id="uniform_diff_0.5",
    ),
    # Orthogonal vectors
    pytest.param(
        "orthogonal",
        lambda: torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        lambda t: torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16),
        None,  # max: skip expected check
        None,  # mean: skip expected check
        id="orthogonal_vectors",
    ),
    # Opposite vectors
    pytest.param(
        "opposite",
        lambda: torch.ones(1, 16, dtype=torch.bfloat16),
        lambda t: -torch.ones(1, 16, dtype=torch.bfloat16),
        None,  # max: skip expected check
        None,  # mean: skip expected check
        id="opposite_vectors",
    ),
    # Large tensors
    pytest.param(
        "large_128x256",
        lambda: torch.randn(128, 256, dtype=torch.bfloat16),
        lambda t: t + 0.1,
        None,  # max: skip expected check
        None,  # mean: skip expected check
        id="large_128x256",
    ),
    # Edge cases - all zeros
    pytest.param(
        "all_zeros",
        lambda: torch.zeros(16, 16, dtype=torch.bfloat16),
        lambda t: t.clone(),
        (0.0, 1e-6),  # max: expect 0
        (0.0, 1e-6),  # mean: expect 0
        id="edge_all_zeros",
    ),
    # Edge cases - all ones
    pytest.param(
        "all_ones",
        lambda: torch.ones(16, 16, dtype=torch.bfloat16),
        lambda t: t.clone(),
        (0.0, 1e-6),  # max: expect 0
        (0.0, 1e-6),  # mean: expect 0
        id="edge_all_ones",
    ),
]


@pytest.mark.parametrize("name,tensor_a_fn,tensor_b_fn,max_spec,mean_spec", TEST_CASES)
def test_abs_metrics_vs_pytorch(ttnn_mesh_device, layout, dtype, name, tensor_a_fn, tensor_b_fn, max_spec, mean_spec):
    """
    Unified test for all metrics against PyTorch ground truth.
    Tests various tensor configurations and verifies TTNN metrics match PyTorch.
    Each metric spec is (expected_value, tolerance) or None.
    """
    # Generate tensors
    torch_a = tensor_a_fn()
    torch_b = tensor_b_fn(torch_a) if callable(tensor_b_fn) else tensor_b_fn

    # Compute PyTorch ground truth (use TTNN-like quantized tensors for quantized dtypes)
    torch_a_q = _quantize_like_ttnn(torch_a, ttnn_mesh_device, dtype, layout)
    torch_b_q = _quantize_like_ttnn(torch_b, ttnn_mesh_device, dtype, layout)
    max_error_torch = (torch_a_q - torch_b_q).abs().max().item()
    mean_error_torch = (torch_a_q - torch_b_q).abs().mean().item()

    # Convert to TTNN
    ttnn_a = ttnn.from_torch(torch_a, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    ttnn_b = ttnn.from_torch(torch_b, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    # Compute with TTNN (ttnn vs ttnn)
    max_error_ttnn = compute_max_abs_error(ttnn_a, ttnn_b)
    mean_error_ttnn = compute_mean_abs_error(ttnn_a, ttnn_b)

    # Mixed-mode checks (torch vs ttnn and ttnn vs torch)
    # Use quantized torch views for mixed checks to fairly compare quantized dtypes
    max_error_mixed_torch_ttnn = compute_max_abs_error(torch_a_q, ttnn_b)
    mean_error_mixed_torch_ttnn = compute_mean_abs_error(torch_a_q, ttnn_b)
    max_error_mixed_ttnn_torch = compute_max_abs_error(ttnn_a, torch_b_q)
    mean_error_mixed_ttnn_torch = compute_mean_abs_error(ttnn_a, torch_b_q)

    # Default tolerance for TTNN vs PyTorch comparison (bf16 precision)
    default_tolerance = 0.02

    # Verify max_abs_error
    if max_spec is not None:
        expected_max, tolerance_max = max_spec
        assert (
            abs(max_error_ttnn - expected_max) < tolerance_max
        ), f"max_abs_error: expected {expected_max}, got {max_error_ttnn}"
        # Check TTNN matches PyTorch
        assert (
            abs(max_error_ttnn - max_error_torch) < tolerance_max
        ), f"max_abs_error TTNN vs PyTorch: {max_error_ttnn} vs {max_error_torch}"
        # Mixed variants also match expected and PyTorch
        assert (
            abs(max_error_mixed_torch_ttnn - expected_max) < tolerance_max
        ), f"max_abs_error (torch,ttnn): expected {expected_max}, got {max_error_mixed_torch_ttnn}"
        assert (
            abs(max_error_mixed_ttnn_torch - expected_max) < tolerance_max
        ), f"max_abs_error (ttnn,torch): expected {expected_max}, got {max_error_mixed_ttnn_torch}"
        assert (
            abs(max_error_mixed_torch_ttnn - max_error_torch) < tolerance_max
        ), f"max_abs_error (torch,ttnn) vs PyTorch: {max_error_mixed_torch_ttnn} vs {max_error_torch}"
        assert (
            abs(max_error_mixed_ttnn_torch - max_error_torch) < tolerance_max
        ), f"max_abs_error (ttnn,torch) vs PyTorch: {max_error_mixed_ttnn_torch} vs {max_error_torch}"
    else:
        # No expected value, just check TTNN matches PyTorch with default tolerance
        assert (
            abs(max_error_ttnn - max_error_torch) < default_tolerance
        ), f"max_abs_error TTNN vs PyTorch: {max_error_ttnn} vs {max_error_torch}"
        assert (
            abs(max_error_mixed_torch_ttnn - max_error_torch) < default_tolerance
        ), f"max_abs_error (torch,ttnn) vs PyTorch: {max_error_mixed_torch_ttnn} vs {max_error_torch}"
        assert (
            abs(max_error_mixed_ttnn_torch - max_error_torch) < default_tolerance
        ), f"max_abs_error (ttnn,torch) vs PyTorch: {max_error_mixed_ttnn_torch} vs {max_error_torch}"

    # Verify mean_abs_error
    if mean_spec is not None:
        expected_mean, tolerance_mean = mean_spec
        assert (
            abs(mean_error_ttnn - expected_mean) < tolerance_mean
        ), f"mean_abs_error: expected {expected_mean}, got {mean_error_ttnn}"
        # Check TTNN matches PyTorch
        assert (
            abs(mean_error_ttnn - mean_error_torch) < tolerance_mean
        ), f"mean_abs_error TTNN vs PyTorch: {mean_error_ttnn} vs {mean_error_torch}"
        # Mixed variants also match expected and PyTorch
        assert (
            abs(mean_error_mixed_torch_ttnn - expected_mean) < tolerance_mean
        ), f"mean_abs_error (torch,ttnn): expected {expected_mean}, got {mean_error_mixed_torch_ttnn}"
        assert (
            abs(mean_error_mixed_ttnn_torch - expected_mean) < tolerance_mean
        ), f"mean_abs_error (ttnn,torch): expected {expected_mean}, got {mean_error_mixed_ttnn_torch}"
        assert (
            abs(mean_error_mixed_torch_ttnn - mean_error_torch) < tolerance_mean
        ), f"mean_abs_error (torch,ttnn) vs PyTorch: {mean_error_mixed_torch_ttnn} vs {mean_error_torch}"
        assert (
            abs(mean_error_mixed_ttnn_torch - mean_error_torch) < tolerance_mean
        ), f"mean_abs_error (ttnn,torch) vs PyTorch: {mean_error_mixed_ttnn_torch} vs {mean_error_torch}"
    else:
        # No expected value, just check TTNN matches PyTorch with default tolerance
        assert (
            abs(mean_error_ttnn - mean_error_torch) < default_tolerance
        ), f"mean_abs_error TTNN vs PyTorch: {mean_error_ttnn} vs {mean_error_torch}"
        assert (
            abs(mean_error_mixed_torch_ttnn - mean_error_torch) < default_tolerance
        ), f"mean_abs_error (torch,ttnn) vs PyTorch: {mean_error_mixed_torch_ttnn} vs {mean_error_torch}"
        assert (
            abs(mean_error_mixed_ttnn_torch - mean_error_torch) < default_tolerance
        ), f"mean_abs_error (ttnn,torch) vs PyTorch: {mean_error_mixed_ttnn_torch} vs {mean_error_torch}"


# check pcc computed on device against known good pcc values on host
def test_pcc_ttnn_native(ttnn_mesh_device, layout, dtype):
    """
    Test TTNN-native PCC computation with actual TTNN tensors on device.

    Verifies that:
    1. PCC computes correctly on-device using TTNN ops
    2. Results match the robust CPU/numpy implementation
    3. Handles various correlation patterns (perfect, high, negative)
    """
    print("\nTest: TTNN-native PCC computation")

    # Test case 1: Perfect positive correlation
    torch.manual_seed(42)
    a_torch = torch.randn(32, 32).bfloat16()
    b_torch = a_torch.clone()

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    # Mixed-mode PCC (ttnn vs torch and torch vs ttnn)
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  Perfect correlation - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn >= 0.999, f"Perfect correlation should be ~1.0, got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.01, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"
    # Mixed should match PyTorch within same tolerance
    assert abs(pcc_mixed_1 - pcc_torch) < 0.01, f"PCC (ttnn,torch) vs PyTorch mismatch: {pcc_mixed_1} vs {pcc_torch}"
    assert abs(pcc_mixed_2 - pcc_torch) < 0.01, f"PCC (torch,ttnn) vs PyTorch mismatch: {pcc_mixed_2} vs {pcc_torch}"

    # Test case 2: High correlation with small noise
    torch.manual_seed(42)
    a_torch = torch.randn(32, 64).bfloat16()
    b_torch = a_torch + torch.randn(32, 64).bfloat16() * 0.01

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  High correlation - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn >= 0.95, f"High correlation should be >= 0.95, got {pcc_ttnn}"
    # Allow more tolerance here due to bfloat16 and noise
    assert abs(pcc_ttnn - pcc_torch) < 0.05, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"
    assert abs(pcc_mixed_1 - pcc_torch) < 0.05, f"PCC (ttnn,torch) vs PyTorch mismatch: {pcc_mixed_1} vs {pcc_torch}"
    assert abs(pcc_mixed_2 - pcc_torch) < 0.05, f"PCC (torch,ttnn) vs PyTorch mismatch: {pcc_mixed_2} vs {pcc_torch}"

    # Test case 3: Negative correlation
    torch.manual_seed(42)
    a_torch = torch.randn(32, 32).bfloat16()
    b_torch = -a_torch + torch.randn(32, 32).bfloat16() * 0.1

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  Negative correlation - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn < -0.8, f"Negative correlation should be < -0.8, got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.1, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"
    assert abs(pcc_mixed_1 - pcc_torch) < 0.1, f"PCC (ttnn,torch) vs PyTorch mismatch: {pcc_mixed_1} vs {pcc_torch}"
    assert abs(pcc_mixed_2 - pcc_torch) < 0.1, f"PCC (torch,ttnn) vs PyTorch mismatch: {pcc_mixed_2} vs {pcc_torch}"

    # Test case 4: Larger tensor (128x256)
    torch.manual_seed(42)
    a_torch = torch.randn(128, 256).bfloat16()
    b_torch = a_torch + torch.randn(128, 256).bfloat16() * 0.05

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  Large tensor - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn >= 0.90, f"Large tensor PCC should be >= 0.90, got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.1, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"
    assert abs(pcc_mixed_1 - pcc_torch) < 0.1, f"PCC (ttnn,torch) vs PyTorch mismatch: {pcc_mixed_1} vs {pcc_torch}"
    assert abs(pcc_mixed_2 - pcc_torch) < 0.1, f"PCC (torch,ttnn) vs PyTorch mismatch: {pcc_mixed_2} vs {pcc_torch}"

    # Test case 5: Large noise - should produce low PCC (< 1.0)
    torch.manual_seed(42)
    a_torch = torch.randn(32, 32).bfloat16()
    # Add significant noise (50% of signal strength)
    b_torch = a_torch + torch.randn(32, 32).bfloat16() * 0.5

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  Large noise (0.5x) - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn < 0.99, f"Large noise should reduce PCC below 0.99, got {pcc_ttnn}"
    assert pcc_ttnn > 0.50, f"PCC should still show some correlation (>0.50), got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.1, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"
    assert abs(pcc_mixed_1 - pcc_torch) < 0.1, f"PCC (ttnn,torch) vs PyTorch mismatch: {pcc_mixed_1} vs {pcc_torch}"
    assert abs(pcc_mixed_2 - pcc_torch) < 0.1, f"PCC (torch,ttnn) vs PyTorch mismatch: {pcc_mixed_2} vs {pcc_torch}"

    # Test case 6: Very large noise - should produce very low PCC
    torch.manual_seed(42)
    a_torch = torch.randn(32, 32).bfloat16()
    # Add massive noise (2x signal strength) - correlation should be weak
    b_torch = a_torch + torch.randn(32, 32).bfloat16() * 2.0

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  Very large noise (2.0x) - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn < 0.80, f"Very large noise should reduce PCC below 0.80, got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.15, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"
    assert abs(pcc_mixed_1 - pcc_torch) < 0.15, f"PCC (ttnn,torch) vs PyTorch mismatch: {pcc_mixed_1} vs {pcc_torch}"
    assert abs(pcc_mixed_2 - pcc_torch) < 0.15, f"PCC (torch,ttnn) vs PyTorch mismatch: {pcc_mixed_2} vs {pcc_torch}"

    print("  ✓ TTNN-native PCC correctly detects varying correlation strengths!")


def test_pcc_constant_tensors(ttnn_mesh_device, layout, dtype):
    """
    Test that TTNN-native PCC correctly handles constant tensors.

    Constant tensors are an edge case where variance is zero. The implementation
    should detect this and return 1.0 if both constants are equal, 0.0 otherwise.
    """
    print("\nTest: PCC with constant tensors")

    # Test case 1: Same constant value
    a_torch = torch.ones(32, 32).bfloat16() * 5.0
    b_torch = torch.ones(32, 32).bfloat16() * 5.0

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    # Mixed-mode PCC computation
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  Same constant (5.0) - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn == 1.0, f"Same constant tensors should have PCC=1.0, got {pcc_ttnn}"
    assert pcc_torch == 1.0, f"PyTorch should also return 1.0, got {pcc_torch}"
    assert pcc_mixed_1 == pcc_torch, f"PCC (ttnn,torch) should equal PyTorch: {pcc_mixed_1} vs {pcc_torch}"
    assert pcc_mixed_2 == pcc_torch, f"PCC (torch,ttnn) should equal PyTorch: {pcc_mixed_2} vs {pcc_torch}"

    # Test case 2: Different constant values
    a_torch = torch.ones(32, 32).bfloat16() * 5.0
    b_torch = torch.ones(32, 32).bfloat16() * 3.0

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  Different constants (5.0 vs 3.0) - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn == 0.0, f"Different constant tensors should have PCC=0.0, got {pcc_ttnn}"
    # PyTorch returns True/False which becomes 1.0/0.0
    assert pcc_torch in [0.0, 1.0], f"PyTorch should return 0.0 or 1.0, got {pcc_torch}"
    assert pcc_mixed_1 == pcc_torch, f"PCC (ttnn,torch) should equal PyTorch: {pcc_mixed_1} vs {pcc_torch}"
    assert pcc_mixed_2 == pcc_torch, f"PCC (torch,ttnn) should equal PyTorch: {pcc_mixed_2} vs {pcc_torch}"

    # Test case 3: All zeros
    a_torch = torch.zeros(32, 32).bfloat16()
    b_torch = torch.zeros(32, 32).bfloat16()

    a_ttnn = ttnn.from_torch(a_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_ttnn = ttnn.from_torch(b_torch, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    a_q = _quantize_like_ttnn(a_torch, ttnn_mesh_device, dtype, layout)
    b_q = _quantize_like_ttnn(b_torch, ttnn_mesh_device, dtype, layout)
    pcc_torch = compute_pcc(a_q, b_q)
    pcc_mixed_1 = compute_pcc(a_ttnn, b_q)
    pcc_mixed_2 = compute_pcc(a_q, b_ttnn)

    print(f"  All zeros - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn == 1.0, f"Zero tensors should have PCC=1.0, got {pcc_ttnn}"
    assert pcc_torch == 1.0, f"PyTorch should also return 1.0, got {pcc_torch}"
    assert pcc_mixed_1 == pcc_torch, f"PCC (ttnn,torch) should equal PyTorch: {pcc_mixed_1} vs {pcc_torch}"
    assert pcc_mixed_2 == pcc_torch, f"PCC (torch,ttnn) should equal PyTorch: {pcc_mixed_2} vs {pcc_torch}"

    print("  ✓ TTNN-native PCC correctly handles constant tensors!")


def test_pcc_all_nan_and_mixed_nan(ttnn_mesh_device, layout, dtype):
    """
    TTNN-native PCC should mirror CPU semantics for NaN cases:
    - both all-NaN -> 1.0
    - mixed NaN presence -> 0.0
    """
    print("\nTest: PCC NaN edge cases")

    a_nan = torch.full((32, 32), float("nan"), dtype=torch.float32)
    b_nan = torch.full((32, 32), float("nan"), dtype=torch.float32)
    a_num = torch.zeros(32, 32, dtype=torch.float32)

    a_nan_t = ttnn.from_torch(a_nan, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_nan_t = ttnn.from_torch(b_nan, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    a_num_t = ttnn.from_torch(a_num, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    # Both all-NaN -> 1.0
    pcc_ttnn = compute_pcc(a_nan_t, b_nan_t)
    a_nan_q = _quantize_like_ttnn(a_nan, ttnn_mesh_device, dtype, layout)
    b_nan_q = _quantize_like_ttnn(b_nan, ttnn_mesh_device, dtype, layout)
    pcc_cpu = compute_pcc(a_nan_q, b_nan_q)
    # Mixed-mode PCC computation
    pcc_mixed_1 = compute_pcc(a_nan_t, b_nan_q)
    pcc_mixed_2 = compute_pcc(a_nan_q, b_nan_t)
    print(f"  both NaN - TTNN: {pcc_ttnn}, CPU: {pcc_cpu}")
    assert pcc_ttnn == 1.0
    assert pcc_cpu == 1.0
    assert pcc_mixed_1 == 1.0
    assert pcc_mixed_2 == 1.0

    # Mixed NaN presence -> 0.0
    pcc_ttnn = compute_pcc(a_nan_t, a_num_t)
    a_num_q = _quantize_like_ttnn(a_num, ttnn_mesh_device, dtype, layout)
    pcc_cpu = compute_pcc(a_nan_q, a_num_q)
    # Mixed-mode PCC computation
    pcc_mixed_1 = compute_pcc(a_nan_t, a_num_q)
    pcc_mixed_2 = compute_pcc(a_nan_q, a_num_t)
    print(f"  mixed NaN - TTNN: {pcc_ttnn}, CPU: {pcc_cpu}")
    assert pcc_ttnn == 0.0
    assert pcc_cpu == 0.0
    assert pcc_mixed_1 == 0.0
    assert pcc_mixed_2 == 0.0

    print("  ✓ TTNN-native PCC correctly handles NaN cases!")


def test_pcc_zero_vs_nonzero(ttnn_mesh_device, layout, dtype):
    """
    One tensor all-zero and the other non-zero -> PCC = 0.0 on both TTNN and CPU.
    """
    print("\nTest: PCC zero vs non-zero")

    a_zero = torch.zeros(32, 32, dtype=torch.float32)
    b_nonzero = torch.ones(32, 32, dtype=torch.float32)

    a_zero_t = ttnn.from_torch(a_zero, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_nonzero_t = ttnn.from_torch(b_nonzero, device=ttnn_mesh_device, dtype=dtype, layout=layout)

    pcc_ttnn = compute_pcc(a_zero_t, b_nonzero_t)
    a_zero_q = _quantize_like_ttnn(a_zero, ttnn_mesh_device, dtype, layout)
    b_nonzero_q = _quantize_like_ttnn(b_nonzero, ttnn_mesh_device, dtype, layout)
    pcc_cpu = compute_pcc(a_zero_q, b_nonzero_q)
    # Mixed-mode PCC computation
    pcc_mixed_1 = compute_pcc(a_zero_t, b_nonzero_q)
    pcc_mixed_2 = compute_pcc(a_zero_q, b_nonzero_t)
    print(f"  zero vs non-zero - TTNN: {pcc_ttnn}, CPU: {pcc_cpu}")
    assert pcc_ttnn == 0.0
    assert pcc_cpu == 0.0
    assert pcc_mixed_1 == 0.0
    assert pcc_mixed_2 == 0.0
    print("  ✓ TTNN-native PCC correctly handles zero vs non-zero!")


def test_comp_allclose_ttnn_native(ttnn_mesh_device, layout, dtype):
    """TTNN-native tests for comp_allclose using on-device ops."""

    # Exact equality
    a_t = ttnn.from_torch(torch.randn(2, 4, dtype=torch.bfloat16), device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_t = a_t
    passed, msg = comp_allclose(a_t, b_t)
    assert passed, f"TTNN equality should pass. Got: {msg}"

    # Fail with tight tolerance
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    b = a + 0.5
    a_t = ttnn.from_torch(a, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_t = ttnn.from_torch(b, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    passed, msg = comp_allclose(a_t, b_t, rtol=1e-6, atol=1e-6)
    assert not passed and "Allclose check failed" in msg

    # Pass with relaxed tolerance
    passed, msg = comp_allclose(a_t, b_t, rtol=0.2, atol=0.6)
    assert passed, f"Expected pass with relaxed tolerance. Got: {msg}"

    # NaN equal
    a = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
    b = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
    a_t = ttnn.from_torch(a, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_t = ttnn.from_torch(b, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    passed, _ = comp_allclose(a_t, b_t)
    assert passed, "TTNN: Both NaNs at same positions should pass"

    # Inf same sign pass, different sign fail
    a = torch.tensor([float("inf"), -float("inf"), 2.0], dtype=torch.float32)
    b = torch.tensor([float("inf"), -float("inf"), 2.0], dtype=torch.float32)
    a_t = ttnn.from_torch(a, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    b_t = ttnn.from_torch(b, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    passed, _ = comp_allclose(a_t, b_t)
    assert passed

    b = torch.tensor([float("inf"), float("inf"), 2.0], dtype=torch.float32)
    b_t = ttnn.from_torch(b, device=ttnn_mesh_device, dtype=dtype, layout=layout)
    passed, msg = comp_allclose(a_t, b_t)
    assert not passed and "Allclose check failed" in msg


if __name__ == "__main__":
    # Run with pytest for better output
    pytest.main([__file__, "-v", "--tb=short"])
