#!/usr/bin/env python3
"""
Test the new match_signature=True feature.
Simple test using PyTorch tensors without mocking.
"""

import torch

from tt_transformers_v2.src.testing.validate_against import (
    clear_validation_results,
    get_validation_registry,
    validate_against,
)


def test_match_signature_method():
    """Test match_signature with a method using PyTorch tensors"""
    print("Test 1: match_signature with method")
    clear_validation_results()

    class SimpleLayer:
        def __init__(self, scale):
            self.scale = scale

        def _reference_impl(self, x):
            """Reference with same signature as __call__"""
            return x * self.scale

        @validate_against(
            reference_fn=lambda self, x: self._reference_impl(x),
            match_signature=True,  # Key feature!
            tolerances={"max_abs_error": 1e-6, "pcc": 0.99},  # PCC check
        )
        def __call__(self, x):
            # Implementation - same as reference
            return x * self.scale

    layer = SimpleLayer(scale=2.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    result = layer(x)

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed, "Validation should pass"

    # Check that PCC is high (close to 1.0)
    assert "pcc" in registry.results[0].metrics
    assert (
        registry.results[0].metrics["pcc"] >= 0.99
    ), f"PCC should be >= 0.99, got {registry.results[0].metrics['pcc']}"

    print("  ✓ match_signature works with methods!")
    print(f"  Metrics: {registry.results[0].metrics}")


def test_match_signature_vs_input_map():
    """Compare match_signature vs input_map - both should work"""
    print("\nTest 2: match_signature vs input_map comparison")
    clear_validation_results()

    # Pattern 1: match_signature
    class LayerWithMatchSignature:
        def __init__(self, scale):
            self.scale = scale

        def _reference(self, x):
            return x * self.scale

        @validate_against(
            reference_fn=lambda self, x: self._reference(x),
            match_signature=True,
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, x):
            return x * self.scale

    # Pattern 2: input_map
    class LayerWithInputMap:
        def __init__(self, scale):
            self.scale = scale

        @validate_against(
            reference_fn=lambda x, scale: x * scale,
            input_map=lambda args, kwargs: ((args[1], args[0].scale), {}),
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, x):
            return x * self.scale

    # Test both
    x = torch.tensor([1.0, 2.0, 3.0])

    layer1 = LayerWithMatchSignature(scale=3.0)
    result1 = layer1(x)

    layer2 = LayerWithInputMap(scale=3.0)
    result2 = layer2(x)

    registry = get_validation_registry()
    assert len(registry.results) == 2
    assert registry.results[0].passed, "match_signature should pass"
    assert registry.results[1].passed, "input_map should pass"
    print("  ✓ Both patterns work correctly!")


def test_match_signature_multi_args():
    """Test match_signature with multiple arguments"""
    print("\nTest 3: match_signature with multiple args")
    clear_validation_results()

    class MultiArgLayer:
        def _reference(self, a, b, c):
            """Reference with same signature"""
            return a * b + c

        @validate_against(
            reference_fn=lambda self, a, b, c: self._reference(a, b, c),
            match_signature=True,
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, a, b, c):
            return a * b + c

    layer = MultiArgLayer()
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    c = 5.0

    result = layer(a, b, c)

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Multiple args work!")


def test_match_signature_with_kwargs():
    """Test match_signature with keyword arguments"""
    print("\nTest 4: match_signature with kwargs")
    clear_validation_results()

    class LayerWithKwargs:
        def _reference(self, x, scale=1.0, offset=0.0):
            return x * scale + offset

        @validate_against(
            reference_fn=lambda self, x, scale=1.0, offset=0.0: self._reference(x, scale, offset),
            match_signature=True,
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, x, scale=1.0, offset=0.0):
            return x * scale + offset

    layer = LayerWithKwargs()
    x = torch.tensor([1.0, 2.0, 3.0])

    result = layer(x, scale=2.0, offset=10.0)

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Kwargs work!")


def test_pcc_metric():
    """Test PCC metric specifically"""
    print("\nTest 5: PCC metric validation")
    clear_validation_results()

    class LayerWithPCC:
        def _reference(self, x):
            # Add small noise to test PCC tolerance
            return x + torch.randn_like(x) * 0.01

        @validate_against(
            reference_fn=lambda self, x: self._reference(x),
            match_signature=True,
            tolerances={"pcc": 0.95},  # Lower threshold due to noise
        )
        def __call__(self, x):
            return x

    torch.manual_seed(42)  # For reproducibility
    layer = LayerWithPCC()
    x = torch.randn(100)
    result = layer(x)

    registry = get_validation_registry()
    assert len(registry.results) == 1

    r = registry.results[0]
    print(f"  PCC value: {r.metrics['pcc']:.4f}")
    # PCC should be high but not perfect due to added noise
    assert 0.90 < r.metrics["pcc"] < 1.0, f"PCC should be in range (0.90, 1.0), got {r.metrics['pcc']}"
    print("  ✓ PCC metric works correctly!")


if __name__ == "__main__":
    test_match_signature_method()
    test_match_signature_vs_input_map()
    test_match_signature_multi_args()
    test_match_signature_with_kwargs()
    test_pcc_metric()
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
