#!/usr/bin/env python3
"""
Test TTNN-native metrics without torch conversion.
"""

import sys

import torch


# Mock TTNN for testing
class MockTTNNTensor:
    """Base class for mock TTNN tensors"""


class MockTTNN(MockTTNNTensor):
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"MockTTNN({self.data})"


def mock_ttnn_to_torch(x, *args, **kwargs):
    if isinstance(x, MockTTNN):
        return x.data
    return x


def mock_ttnn_subtract(a, b, *args, **kwargs):
    return MockTTNN(a.data - b.data)


def mock_ttnn_abs(x, *args, **kwargs):
    return MockTTNN(x.data.abs())


def mock_ttnn_max(x, *args, **kwargs):
    return MockTTNN(x.data.max())


def mock_ttnn_mean(x, *args, **kwargs):
    return MockTTNN(x.data.mean())


# Mock TTNN module with Tensor class
sys.modules["ttnn"] = type(
    "MockModule",
    (),
    {
        "Tensor": MockTTNNTensor,  # isinstance checks will work!
        "to_torch": staticmethod(mock_ttnn_to_torch),
        "subtract": staticmethod(mock_ttnn_subtract),
        "abs": staticmethod(mock_ttnn_abs),
        "max": staticmethod(mock_ttnn_max),
        "mean": staticmethod(mock_ttnn_mean),
    },
)()

from tt_transformers_v2.src.testing.validation import (
    clear_validation_results,
    get_validation_registry,
    validate_against,
)


def test_ttnn_native_metrics():
    """Test that metrics work directly on TTNN tensors"""
    print("Test: TTNN-native metrics")
    clear_validation_results()

    class SimpleLayer:
        def _reference_impl(self, x):
            """Reference returns TTNN"""
            x_torch = mock_ttnn_to_torch(x)
            result_torch = x_torch * 2.0
            return MockTTNN(result_torch)  # Return TTNN

        @validate_against(
            reference_fn=lambda self, x: self._reference_impl(x),
            match_signature=True,
            # No conversions! Metrics computed on TTNN directly
            tolerances={"max_abs_error": 1e-5},
        )
        def __call__(self, x):
            """Implementation returns TTNN"""
            x_torch = mock_ttnn_to_torch(x)
            result_torch = x_torch * 2.0
            return MockTTNN(result_torch)  # Return TTNN

    layer = SimpleLayer()
    x = MockTTNN(torch.tensor([1.0, 2.0, 3.0]))
    result = layer(x)

    registry = get_validation_registry()
    assert len(registry.results) == 1, f"Expected 1 result, got {len(registry.results)}"

    # Check results
    r = registry.results[0]
    assert r.passed, f"Validation should pass: {r.errors}"
    assert "max_abs_error" in r.metrics
    assert "mean_abs_error" in r.metrics

    print(f"  ✓ Metrics computed on TTNN tensors:")
    print(f"    max_abs_error: {r.metrics['max_abs_error']}")
    print(f"    mean_abs_error: {r.metrics['mean_abs_error']}")
    print(f"  ✓ No conversions needed in decorator!")


def test_torch_metrics_still_work():
    """Test that torch tensors still work"""
    print("\nTest: Torch tensor metrics still work")
    clear_validation_results()

    @validate_against(reference_fn=lambda x: x * 2.0, match_signature=True, tolerances={"max_abs_error": 1e-6})
    def my_function(x):
        return x * 2.0

    result = my_function(torch.tensor([1.0, 2.0, 3.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Torch tensors still work")


def test_mixed_no_conversion():
    """Test that TTNN → TTNN requires minimal conversions"""
    print("\nTest: TTNN → TTNN with minimal conversions")
    clear_validation_results()

    class Layer:
        def _reference(self, x):
            # Internal conversion for computation is OK
            temp_torch = mock_ttnn_to_torch(x)
            result_torch = temp_torch * 3.0
            return MockTTNN(result_torch)  # Return TTNN

        @validate_against(
            reference_fn=lambda self, x: self._reference(x), match_signature=True, tolerances={"max_abs_error": 1e-5}
        )
        def __call__(self, x):
            temp_torch = mock_ttnn_to_torch(x)
            result_torch = temp_torch * 3.0
            return MockTTNN(result_torch)  # Return TTNN

    layer = Layer()
    x = MockTTNN(torch.tensor([1.0, 2.0]))
    result = layer(x)

    registry = get_validation_registry()
    r = registry.results[0]
    print(f"  ✓ Validation: {'PASS' if r.passed else 'FAIL'}")
    print(f"  ✓ Metrics computed on TTNN tensors directly!")
    assert r.passed


def run_all_tests():
    print("=" * 80)
    print("TTNN-NATIVE METRICS TESTS")
    print("=" * 80)

    tests = [
        test_ttnn_native_metrics,
        test_torch_metrics_still_work,
        test_mixed_no_conversion,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"TESTS COMPLETE: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
