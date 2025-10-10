#!/usr/bin/env python3
"""
Test the new match_signature=True feature.
Simple test without requiring TTNN hardware.
"""

import sys

import torch


# Mock TTNN
class MockTTNN:
    def __init__(self, data):
        self.data = data


def mock_ttnn_to_torch(x):
    if isinstance(x, MockTTNN):
        return x.data
    return x


sys.modules["ttnn"] = type("MockModule", (), {"to_torch": mock_ttnn_to_torch})()

from ds_r1_qwen import clear_validation_results, get_validation_registry, validate_against


def test_match_signature_method():
    """Test match_signature with a method"""
    print("Test 1: match_signature with method")
    clear_validation_results()

    class SimpleLayer:
        def __init__(self, scale):
            self.scale = scale

        def _reference_impl(self, x):
            """Reference with same signature as __call__"""
            # Convert mock TTNN to torch
            x_torch = mock_ttnn_to_torch(x)
            return x_torch * self.scale

        @validate_against(
            reference_fn=lambda self, x: self._reference_impl(x),
            match_signature=True,  # Key feature!
            output_map_impl=lambda x: mock_ttnn_to_torch(x),
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, x):
            # "TTNN" implementation
            x_torch = mock_ttnn_to_torch(x)
            result = x_torch * self.scale
            return MockTTNN(result)

    layer = SimpleLayer(scale=2.0)
    x = MockTTNN(torch.tensor([1.0, 2.0, 3.0]))
    result = layer(x)

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed, "Validation should pass"
    print("  ✓ match_signature with method works!")


def test_match_signature_vs_input_map():
    """Compare match_signature vs input_map - both should work"""
    print("\nTest 2: match_signature vs input_map comparison")
    clear_validation_results()

    # Pattern 1: match_signature
    class LayerWithMatchSignature:
        def __init__(self, scale):
            self.scale = scale

        def _reference(self, x):
            x_torch = mock_ttnn_to_torch(x)
            return x_torch * self.scale

        @validate_against(
            reference_fn=lambda self, x: self._reference(x),
            match_signature=True,
            output_map_impl=lambda x: mock_ttnn_to_torch(x),
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, x):
            x_torch = mock_ttnn_to_torch(x)
            return MockTTNN(x_torch * self.scale)

    # Pattern 2: input_map
    class LayerWithInputMap:
        def __init__(self, scale):
            self.scale = scale

        @validate_against(
            reference_fn=lambda x, scale: x * scale,
            input_map=lambda args, kwargs: ((mock_ttnn_to_torch(args[1]), args[0].scale), {}),
            output_map_impl=lambda x: mock_ttnn_to_torch(x),
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, x):
            x_torch = mock_ttnn_to_torch(x)
            return MockTTNN(x_torch * self.scale)

    # Test both
    x = MockTTNN(torch.tensor([1.0, 2.0, 3.0]))

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
            a_t = mock_ttnn_to_torch(a)
            b_t = mock_ttnn_to_torch(b)
            return a_t * b_t + c

        @validate_against(
            reference_fn=lambda self, a, b, c: self._reference(a, b, c),
            match_signature=True,
            output_map_impl=lambda x: mock_ttnn_to_torch(x),
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, a, b, c):
            a_t = mock_ttnn_to_torch(a)
            b_t = mock_ttnn_to_torch(b)
            return MockTTNN(a_t * b_t + c)

    layer = MultiArgLayer()
    a = MockTTNN(torch.tensor([1.0, 2.0]))
    b = MockTTNN(torch.tensor([3.0, 4.0]))
    c = 5.0

    result = layer(a, b, c)

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Multiple args with match_signature works!")


def test_match_signature_with_kwargs():
    """Test match_signature with keyword arguments"""
    print("\nTest 4: match_signature with kwargs")
    clear_validation_results()

    class LayerWithKwargs:
        def _reference(self, x, scale=1.0, offset=0.0):
            x_t = mock_ttnn_to_torch(x)
            return x_t * scale + offset

        @validate_against(
            reference_fn=lambda self, x, scale=1.0, offset=0.0: self._reference(x, scale, offset),
            match_signature=True,
            output_map_impl=lambda x: mock_ttnn_to_torch(x),
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, x, scale=1.0, offset=0.0):
            x_t = mock_ttnn_to_torch(x)
            return MockTTNN(x_t * scale + offset)

    layer = LayerWithKwargs()
    x = MockTTNN(torch.tensor([1.0, 2.0, 3.0]))

    result = layer(x, scale=2.0, offset=10.0)

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Kwargs with match_signature works!")


def run_all_tests():
    print("=" * 80)
    print("MATCH_SIGNATURE FEATURE TESTS")
    print("=" * 80)

    tests = [
        test_match_signature_method,
        test_match_signature_vs_input_map,
        test_match_signature_multi_args,
        test_match_signature_with_kwargs,
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

    if get_validation_registry().results:
        print("\nValidation summary:")
        summary = get_validation_registry().get_summary()
        print(f"  Total validations: {summary['total']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
