#!/usr/bin/env python3
"""
Unit tests for TTNN-native metric functions - NO MOCKS.
Tests numerical correctness of metrics computed with actual TTNN ops.

Run with: pytest test_ttnn_metrics_numerical.py -v
Requires: TTNN hardware/installation, pytest
"""

import pytest
import torch

# Try to import TTNN - skip tests if not available
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

# Pytest skip marker for tests requiring TTNN
pytestmark = pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")

# Import metric functions from the validation framework
from tt_transformers_v2.src.testing import (
    comp_allclose,
    compute_cosine_similarity,
    compute_max_abs_error,
    compute_mean_abs_error,
    compute_pcc,
)


# Pytest fixtures
@pytest.fixture(scope="module")
def ttnn_device():
    """Setup TTNN device for all tests"""
    if not TTNN_AVAILABLE:
        pytest.skip("TTNN not available")

    device_ids = ttnn.get_device_ids()
    if len(device_ids) == 0:
        pytest.skip("No TTNN devices found")

    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
    yield mesh_device
    ttnn.close_mesh_device(mesh_device)


def to_ttnn(torch_tensor, device):
    """Convert PyTorch tensor to TTNN tensor"""
    # TTNN expects [1, 1, ...] shape for tile layout
    while torch_tensor.dim() < 4:
        torch_tensor = torch_tensor.unsqueeze(0)

    return ttnn.from_torch(torch_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


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
        (1.0, 0.01),  # cosine: expect ~1, larger tolerance for bf16
        id="identical_random",
    ),
    # Known differences
    pytest.param(
        "known_diff_0.5",
        lambda: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16),
        lambda t: t + torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=torch.bfloat16),
        (0.5, 0.02),  # max: expect 0.5
        None,  # mean: skip expected check, just compare TTNN vs PyTorch
        None,  # cosine: skip expected check
        id="known_diff_max_0.5",
    ),
    pytest.param(
        "uniform_diff_0.5",
        lambda: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        lambda t: t + 0.5,
        (0.5, 0.02),  # max: expect 0.5
        (0.5, 0.02),  # mean: expect 0.5 (all elements differ by same amount)
        None,  # cosine: skip expected check
        id="uniform_diff_0.5",
    ),
    # Orthogonal vectors
    pytest.param(
        "orthogonal",
        lambda: torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        lambda t: torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16),
        None,  # max: skip expected check
        None,  # mean: skip expected check
        (0.0, 0.1),  # cosine: expect 0, larger tolerance
        id="orthogonal_vectors",
    ),
    # Opposite vectors
    pytest.param(
        "opposite",
        lambda: torch.ones(1, 16, dtype=torch.bfloat16),
        lambda t: -torch.ones(1, 16, dtype=torch.bfloat16),
        None,  # max: skip expected check
        None,  # mean: skip expected check
        (-1.0, 0.02),  # cosine: expect -1
        id="opposite_vectors",
    ),
    # Large tensors
    pytest.param(
        "large_128x256",
        lambda: torch.randn(128, 256, dtype=torch.bfloat16),
        lambda t: t + 0.1,
        None,  # max: skip expected check
        None,  # mean: skip expected check
        None,  # cosine: skip expected check
        id="large_128x256",
    ),
    # Edge cases - all zeros
    pytest.param(
        "all_zeros",
        lambda: torch.zeros(16, 16, dtype=torch.bfloat16),
        lambda t: t.clone(),
        (0.0, 1e-6),  # max: expect 0
        (0.0, 1e-6),  # mean: expect 0
        None,  # cosine: undefined for zero vectors
        id="edge_all_zeros",
    ),
    # Edge cases - all ones
    pytest.param(
        "all_ones",
        lambda: torch.ones(16, 16, dtype=torch.bfloat16),
        lambda t: t.clone(),
        (0.0, 1e-6),  # max: expect 0
        (0.0, 1e-6),  # mean: expect 0
        (1.0, 1e-3),  # cosine: expect 1
        id="edge_all_ones",
    ),
]


@pytest.mark.parametrize("name,tensor_a_fn,tensor_b_fn,max_spec,mean_spec,cosine_spec", TEST_CASES)
def test_metrics_vs_pytorch(ttnn_device, name, tensor_a_fn, tensor_b_fn, max_spec, mean_spec, cosine_spec):
    """
    Unified test for all metrics against PyTorch ground truth.
    Tests various tensor configurations and verifies TTNN metrics match PyTorch.
    Each metric spec is (expected_value, tolerance) or None.
    """
    # Generate tensors
    torch_a = tensor_a_fn()
    torch_b = tensor_b_fn(torch_a) if callable(tensor_b_fn) else tensor_b_fn

    # Compute PyTorch ground truth
    max_error_torch = (torch_a - torch_b).abs().max().item()
    mean_error_torch = (torch_a - torch_b).abs().mean().item()
    cosine_torch = torch.nn.functional.cosine_similarity(torch_a.flatten(), torch_b.flatten(), dim=0).item()

    # Convert to TTNN
    ttnn_a = to_ttnn(torch_a, ttnn_device)
    ttnn_b = to_ttnn(torch_b, ttnn_device)

    # Compute with TTNN
    max_error_ttnn = compute_max_abs_error(ttnn_a, ttnn_b)
    mean_error_ttnn = compute_mean_abs_error(ttnn_a, ttnn_b)
    cosine_ttnn = compute_cosine_similarity(ttnn_a, ttnn_b)

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
    else:
        # No expected value, just check TTNN matches PyTorch with default tolerance
        assert (
            abs(max_error_ttnn - max_error_torch) < default_tolerance
        ), f"max_abs_error TTNN vs PyTorch: {max_error_ttnn} vs {max_error_torch}"

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
    else:
        # No expected value, just check TTNN matches PyTorch with default tolerance
        assert (
            abs(mean_error_ttnn - mean_error_torch) < default_tolerance
        ), f"mean_abs_error TTNN vs PyTorch: {mean_error_ttnn} vs {mean_error_torch}"

    # Verify cosine_similarity
    if cosine_spec is not None:
        expected_cosine, tolerance_cosine = cosine_spec
        assert (
            abs(cosine_ttnn - expected_cosine) < tolerance_cosine
        ), f"cosine_similarity: expected {expected_cosine}, got {cosine_ttnn}"
        # Check TTNN matches PyTorch
        assert (
            abs(cosine_ttnn - cosine_torch) < tolerance_cosine
        ), f"cosine_similarity TTNN vs PyTorch: {cosine_ttnn} vs {cosine_torch}"
    else:
        # No expected value, just check TTNN matches PyTorch with default tolerance
        assert (
            abs(cosine_ttnn - cosine_torch) < default_tolerance
        ), f"cosine_similarity TTNN vs PyTorch: {cosine_ttnn} vs {cosine_torch}"


def test_pcc_ttnn_native(ttnn_device):
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

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  Perfect correlation - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn >= 0.999, f"Perfect correlation should be ~1.0, got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.01, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"

    # Test case 2: High correlation with small noise
    torch.manual_seed(42)
    a_torch = torch.randn(32, 64).bfloat16()
    b_torch = a_torch + torch.randn(32, 64).bfloat16() * 0.01

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  High correlation - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn >= 0.95, f"High correlation should be >= 0.95, got {pcc_ttnn}"
    # Allow more tolerance here due to bfloat16 and noise
    assert abs(pcc_ttnn - pcc_torch) < 0.05, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"

    # Test case 3: Negative correlation
    torch.manual_seed(42)
    a_torch = torch.randn(32, 32).bfloat16()
    b_torch = -a_torch + torch.randn(32, 32).bfloat16() * 0.1

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  Negative correlation - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn < -0.8, f"Negative correlation should be < -0.8, got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.1, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"

    # Test case 4: Larger tensor (128x256)
    torch.manual_seed(42)
    a_torch = torch.randn(128, 256).bfloat16()
    b_torch = a_torch + torch.randn(128, 256).bfloat16() * 0.05

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  Large tensor - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn >= 0.90, f"Large tensor PCC should be >= 0.90, got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.1, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"

    # Test case 5: Large noise - should produce low PCC (< 1.0)
    torch.manual_seed(42)
    a_torch = torch.randn(32, 32).bfloat16()
    # Add significant noise (50% of signal strength)
    b_torch = a_torch + torch.randn(32, 32).bfloat16() * 0.5

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  Large noise (0.5x) - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn < 0.99, f"Large noise should reduce PCC below 0.99, got {pcc_ttnn}"
    assert pcc_ttnn > 0.50, f"PCC should still show some correlation (>0.50), got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.1, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"

    # Test case 6: Very large noise - should produce very low PCC
    torch.manual_seed(42)
    a_torch = torch.randn(32, 32).bfloat16()
    # Add massive noise (2x signal strength) - correlation should be weak
    b_torch = a_torch + torch.randn(32, 32).bfloat16() * 2.0

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  Very large noise (2.0x) - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn < 0.80, f"Very large noise should reduce PCC below 0.80, got {pcc_ttnn}"
    assert abs(pcc_ttnn - pcc_torch) < 0.15, f"TTNN vs PyTorch mismatch: {pcc_ttnn} vs {pcc_torch}"

    print("  ✓ TTNN-native PCC correctly detects varying correlation strengths!")


def test_pcc_constant_tensors(ttnn_device):
    """
    Test that TTNN-native PCC correctly handles constant tensors.

    Constant tensors are an edge case where variance is zero. The implementation
    should detect this and return 1.0 if both constants are equal, 0.0 otherwise.
    """
    print("\nTest: PCC with constant tensors")

    # Test case 1: Same constant value
    a_torch = torch.ones(32, 32).bfloat16() * 5.0
    b_torch = torch.ones(32, 32).bfloat16() * 5.0

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  Same constant (5.0) - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn == 1.0, f"Same constant tensors should have PCC=1.0, got {pcc_ttnn}"
    assert pcc_torch == 1.0, f"PyTorch should also return 1.0, got {pcc_torch}"

    # Test case 2: Different constant values
    a_torch = torch.ones(32, 32).bfloat16() * 5.0
    b_torch = torch.ones(32, 32).bfloat16() * 3.0

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  Different constants (5.0 vs 3.0) - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn == 0.0, f"Different constant tensors should have PCC=0.0, got {pcc_ttnn}"
    # PyTorch returns True/False which becomes 1.0/0.0
    assert pcc_torch in [0.0, 1.0], f"PyTorch should return 0.0 or 1.0, got {pcc_torch}"

    # Test case 3: All zeros
    a_torch = torch.zeros(32, 32).bfloat16()
    b_torch = torch.zeros(32, 32).bfloat16()

    a_ttnn = to_ttnn(a_torch, ttnn_device)
    b_ttnn = to_ttnn(b_torch, ttnn_device)

    pcc_ttnn = compute_pcc(a_ttnn, b_ttnn)
    pcc_torch = compute_pcc(a_torch, b_torch)

    print(f"  All zeros - TTNN: {pcc_ttnn:.6f}, PyTorch: {pcc_torch:.6f}")
    assert pcc_ttnn == 1.0, f"Zero tensors should have PCC=1.0, got {pcc_ttnn}"
    assert pcc_torch == 1.0, f"PyTorch should also return 1.0, got {pcc_torch}"

    print("  ✓ TTNN-native PCC correctly handles constant tensors!")


def test_pcc_all_nan_and_mixed_nan(ttnn_device):
    """
    TTNN-native PCC should mirror CPU semantics for NaN cases:
    - both all-NaN -> 1.0
    - mixed NaN presence -> 0.0
    """
    print("\nTest: PCC NaN edge cases")

    a_nan = torch.full((32, 32), float("nan"), dtype=torch.float32)
    b_nan = torch.full((32, 32), float("nan"), dtype=torch.float32)
    a_num = torch.zeros(32, 32, dtype=torch.float32)

    a_nan_t = to_ttnn(a_nan, ttnn_device)
    b_nan_t = to_ttnn(b_nan, ttnn_device)
    a_num_t = to_ttnn(a_num, ttnn_device)

    # Both all-NaN -> 1.0
    pcc_ttnn = compute_pcc(a_nan_t, b_nan_t)
    pcc_cpu = compute_pcc(a_nan, b_nan)
    print(f"  both NaN - TTNN: {pcc_ttnn}, CPU: {pcc_cpu}")
    assert pcc_ttnn == 1.0
    assert pcc_cpu == 1.0

    # Mixed NaN presence -> 0.0
    pcc_ttnn = compute_pcc(a_nan_t, a_num_t)
    pcc_cpu = compute_pcc(a_nan, a_num)
    print(f"  mixed NaN - TTNN: {pcc_ttnn}, CPU: {pcc_cpu}")
    assert pcc_ttnn == 0.0
    assert pcc_cpu == 0.0

    print("  ✓ TTNN-native PCC correctly handles NaN cases!")


def test_pcc_zero_vs_nonzero(ttnn_device):
    """
    One tensor all-zero and the other non-zero -> PCC = 0.0 on both TTNN and CPU.
    """
    print("\nTest: PCC zero vs non-zero")

    a_zero = torch.zeros(32, 32, dtype=torch.float32)
    b_nonzero = torch.ones(32, 32, dtype=torch.float32)

    a_zero_t = to_ttnn(a_zero, ttnn_device)
    b_nonzero_t = to_ttnn(b_nonzero, ttnn_device)

    pcc_ttnn = compute_pcc(a_zero_t, b_nonzero_t)
    pcc_cpu = compute_pcc(a_zero, b_nonzero)

    print(f"  zero vs non-zero - TTNN: {pcc_ttnn}, CPU: {pcc_cpu}")
    assert pcc_ttnn == 0.0
    assert pcc_cpu == 0.0
    print("  ✓ TTNN-native PCC correctly handles zero vs non-zero!")


# Sanity check: Test metrics with PyTorch tensors (no TTNN device needed)
def test_metrics_pytorch_only():
    """
    Test that metric functions work with pure PyTorch tensors (fallback path).

    Purpose:
    - Validates the PyTorch fallback path in metric functions when TTNN tensors aren't used
    - Provides baseline correctness verification without requiring TTNN hardware
    - Can run in CI/CD environments or on developer machines without accelerators
    - Ensures the fundamental metric logic is sound before testing TTNN-specific paths

    This is the only test that doesn't require TTNN hardware, making it useful for
    quick validation during development and in environments without specialized hardware.
    """
    # This test doesn't require TTNN hardware
    torch_a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    torch_b = torch.tensor([[1.0, 2.0, 3.5], [4.0, 5.0, 6.0]], dtype=torch.float32)

    # Functions should fall back to PyTorch path
    max_error = compute_max_abs_error(torch_a, torch_b)
    mean_error = compute_mean_abs_error(torch_a, torch_b)
    cosine = compute_cosine_similarity(torch_a, torch_b)
    pcc = compute_pcc(torch_a, torch_b)

    # Verify results
    expected_max = 0.5
    expected_mean = (torch_a - torch_b).abs().mean().item()

    assert abs(max_error - expected_max) < 1e-6, f"Expected {expected_max}, got {max_error}"
    assert abs(mean_error - expected_mean) < 1e-6, f"Mean error mismatch"
    assert 0.0 <= cosine <= 1.0, f"Cosine should be in [0,1], got {cosine}"
    assert 0.99 <= pcc <= 1.0, f"PCC should be high (~1.0) for similar tensors, got {pcc}"


def test_comp_allclose_pytorch_only():
    """PyTorch-only tests for comp_allclose covering pass/fail and edge cases."""
    # Exact equality
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = a.clone()
    passed, msg = comp_allclose(a, b)
    assert passed, f"Expected pass for exact equality. Got: {msg}"
    assert "Max ATOL Delta" in msg and "Max RTOL Delta" in msg

    # Within tolerance
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([1.0 + 1e-7, 2.0 - 1e-7, 3.0], dtype=torch.float32)
    passed, msg = comp_allclose(a, b)
    assert passed, f"Expected pass within default tolerance. Got: {msg}"

    # Outside tolerance
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([1.0, 2.1, 3.0], dtype=torch.float32)
    passed, msg = comp_allclose(a, b)
    assert not passed and "Allclose check failed" in msg

    # NaN equality (equal_nan=True semantics)
    a = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    passed, _ = comp_allclose(a, b)
    assert passed, "Both NaNs at same positions should pass"

    # Inf same sign -> pass
    a = torch.tensor([float("inf"), -float("inf"), 1.0])
    b = torch.tensor([float("inf"), -float("inf"), 1.0])
    passed, _ = comp_allclose(a, b)
    assert passed, "Same sign infinities should pass"

    # Inf different sign -> fail
    a = torch.tensor([float("inf"), -float("inf"), 1.0])
    b = torch.tensor([float("inf"), float("inf"), 1.0])
    passed, msg = comp_allclose(a, b)
    assert not passed and "Allclose check failed" in msg


def test_comp_allclose_ttnn_native(ttnn_device):
    """TTNN-native tests for comp_allclose using on-device ops."""

    # Helper to TTNN
    def tt(x):
        return to_ttnn(x, ttnn_device)

    # Exact equality
    a_t = tt(torch.randn(2, 4, dtype=torch.bfloat16))
    b_t = a_t
    passed, msg = comp_allclose(a_t, b_t)
    assert passed, f"TTNN equality should pass. Got: {msg}"

    # Fail with tight tolerance
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    b = a + 0.5
    passed, msg = comp_allclose(tt(a), tt(b), rtol=1e-6, atol=1e-6)
    assert not passed and "Allclose check failed" in msg

    # Pass with relaxed tolerance
    passed, msg = comp_allclose(tt(a), tt(b), rtol=0.2, atol=0.6)
    assert passed, f"Expected pass with relaxed tolerance. Got: {msg}"

    # NaN equal
    a = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
    b = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
    passed, _ = comp_allclose(tt(a), tt(b))
    assert passed, "TTNN: Both NaNs at same positions should pass"

    # Inf same sign pass, different sign fail
    a = torch.tensor([float("inf"), -float("inf"), 2.0], dtype=torch.float32)
    b = torch.tensor([float("inf"), -float("inf"), 2.0], dtype=torch.float32)
    passed, _ = comp_allclose(tt(a), tt(b))
    assert passed

    b = torch.tensor([float("inf"), float("inf"), 2.0], dtype=torch.float32)
    passed, msg = comp_allclose(tt(a), tt(b))
    assert not passed and "Allclose check failed" in msg


if __name__ == "__main__":
    # Run with pytest for better output
    pytest.main([__file__, "-v", "--tb=short"])
