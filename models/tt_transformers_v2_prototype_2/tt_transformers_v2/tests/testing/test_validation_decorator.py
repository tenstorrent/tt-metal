#!/usr/bin/env python3
"""
Unit tests for the validation decorator system.
Tests the decorator without requiring TTNN hardware.
"""

import torch


# Mock TTNN for testing without hardware
class MockTTNN:
    """Mock TTNN tensor for testing"""

    def __init__(self, data):
        self.data = data

    def to_torch(self):
        return self.data


def mock_ttnn_to_torch(x):
    """Mock ttnn.to_torch"""
    if isinstance(x, MockTTNN):
        return x.data
    return x


# Replace ttnn.to_torch for testing
import sys

sys.modules["ttnn"] = type(
    "MockModule",
    (),
    {
        "to_torch": mock_ttnn_to_torch,
    },
)()

from ds_r1_qwen import clear_validation_results, enable_validation, get_validation_registry, validate_against


def test_basic_validation():
    """Test basic validation with simple functions"""
    print("Test 1: Basic validation")
    clear_validation_results()

    # Reference function
    def reference_add(a, b):
        return a + b

    # Implementation with exact match
    @validate_against(reference_fn=reference_add, tolerances={"max_abs_error": 1e-6})
    def impl_add(a, b):
        return a + b

    # Run
    result = impl_add(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

    # Check
    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Basic validation passed")


def test_validation_with_tolerance():
    """Test validation with tolerance checking"""
    print("\nTest 2: Validation with tolerance")
    clear_validation_results()

    def reference_fn(x):
        return x * 2.0

    # Implementation with small error
    @validate_against(reference_fn=reference_fn, tolerances={"max_abs_error": 0.1})
    def impl_fn(x):
        return x * 2.0 + 0.05  # Small error within tolerance

    result = impl_fn(torch.tensor([1.0, 2.0, 3.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Tolerance validation passed")


def test_validation_failure():
    """Test that validation correctly fails"""
    print("\nTest 3: Validation failure detection")
    clear_validation_results()

    def reference_fn(x):
        return x * 2.0

    @validate_against(reference_fn=reference_fn, tolerances={"max_abs_error": 0.01})
    def impl_fn(x):
        return x * 2.0 + 0.5  # Large error, exceeds tolerance

    result = impl_fn(torch.tensor([1.0, 2.0, 3.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert not registry.results[0].passed
    assert len(registry.results[0].errors) > 0
    print("  ✓ Failure detection works")


def test_input_output_mapping():
    """Test input and output mapping functions"""
    print("\nTest 4: Input/output mapping")
    clear_validation_results()

    def reference_fn(x):
        return x * 2.0

    @validate_against(
        reference_fn=reference_fn,
        input_map=lambda args, kwargs: ((args[0].data,), {}),  # Extract .data
        output_map_impl=lambda x: x.data,  # Extract .data
        tolerances={"max_abs_error": 1e-6},
    )
    def impl_fn(wrapped_x):
        return MockTTNN(wrapped_x.data * 2.0)

    wrapped_input = MockTTNN(torch.tensor([1.0, 2.0, 3.0]))
    result = impl_fn(wrapped_input)

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Input/output mapping works")


def test_custom_metrics():
    """Test custom metrics"""
    print("\nTest 5: Custom metrics")
    clear_validation_results()

    def reference_fn(x):
        return x**2

    def relative_error(impl, ref):
        return ((impl - ref).abs() / (ref.abs() + 1e-8)).mean().item()

    @validate_against(
        reference_fn=reference_fn, metrics={"relative_error": relative_error}, tolerances={"relative_error": 0.1}
    )
    def impl_fn(x):
        return x**2

    result = impl_fn(torch.tensor([1.0, 2.0, 3.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    assert "relative_error" in registry.results[0].metrics
    print("  ✓ Custom metrics work")


def test_method_validation():
    """Test validation on class methods"""
    print("\nTest 6: Method validation")
    clear_validation_results()

    class SimpleOp:
        def __init__(self, scale):
            self.scale = scale

        @validate_against(
            reference_fn=lambda x, scale: x * scale,
            input_map=lambda args, kwargs: ((args[1], args[0].scale), {}),
            tolerances={"max_abs_error": 1e-6},
        )
        def __call__(self, x):
            return x * self.scale

    op = SimpleOp(scale=3.0)
    result = op(torch.tensor([1.0, 2.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].passed
    print("  ✓ Method validation works")


def test_enable_disable():
    """Test enabling/disabling validation"""
    print("\nTest 7: Enable/disable validation")
    clear_validation_results()

    def reference_fn(x):
        return x * 2.0

    @validate_against(reference_fn=reference_fn, tolerances={"max_abs_error": 1e-6})
    def impl_fn(x):
        return x * 2.0

    # Disable validation
    enable_validation(False)
    result = impl_fn(torch.tensor([1.0, 2.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 0  # No validation recorded

    # Re-enable
    enable_validation(True)
    result = impl_fn(torch.tensor([1.0, 2.0]))

    assert len(registry.results) == 1  # Validation recorded
    print("  ✓ Enable/disable works")


def test_performance_metrics():
    """Test that performance metrics are collected"""
    print("\nTest 8: Performance metrics")
    clear_validation_results()

    def reference_fn(x):
        return x * 2.0

    @validate_against(reference_fn=reference_fn, performance_metrics=True)
    def impl_fn(x):
        return x * 2.0

    result = impl_fn(torch.tensor([1.0, 2.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert registry.results[0].execution_time_impl > 0
    assert registry.results[0].execution_time_ref > 0
    print("  ✓ Performance metrics collected")


def test_multiple_validations():
    """Test multiple validations and summary"""
    print("\nTest 9: Multiple validations and summary")
    clear_validation_results()

    def ref_add(a, b):
        return a + b

    def ref_mul(a, b):
        return a * b

    @validate_against(reference_fn=ref_add, tolerances={"max_abs_error": 1e-6})
    def add(a, b):
        return a + b

    @validate_against(reference_fn=ref_mul, tolerances={"max_abs_error": 1e-6})
    def mul(a, b):
        return a * b

    # Run multiple validations
    add(torch.tensor([1.0]), torch.tensor([2.0]))
    mul(torch.tensor([3.0]), torch.tensor([4.0]))
    add(torch.tensor([5.0]), torch.tensor([6.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 3

    summary = registry.get_summary()
    assert summary["total"] == 3
    assert summary["passed"] == 3
    assert summary["pass_rate"] == 1.0
    print("  ✓ Multiple validations tracked")


def test_cosine_similarity():
    """Test cosine similarity metric"""
    print("\nTest 10: Cosine similarity metric")
    clear_validation_results()

    def reference_fn(x):
        return x / x.norm()

    @validate_against(
        reference_fn=reference_fn,
        # Default metrics include cosine_similarity
    )
    def impl_fn(x):
        return x / x.norm()

    result = impl_fn(torch.tensor([1.0, 2.0, 3.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert "cosine_similarity" in registry.results[0].metrics
    # Should be very close to 1.0 for identical outputs
    assert registry.results[0].metrics["cosine_similarity"] > 0.99
    print("  ✓ Cosine similarity metric works")


def test_reference_failure_handling():
    """Test handling of reference function failures"""
    print("\nTest 11: Reference failure handling")
    clear_validation_results()

    def failing_reference(x):
        raise RuntimeError("Reference failed!")

    @validate_against(
        reference_fn=failing_reference,
    )
    def impl_fn(x):
        return x * 2.0

    # Should not crash, should log error
    result = impl_fn(torch.tensor([1.0, 2.0]))

    registry = get_validation_registry()
    assert len(registry.results) == 1
    assert not registry.results[0].passed
    assert any("Reference execution failed" in e for e in registry.results[0].errors)
    print("  ✓ Reference failure handled gracefully")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("VALIDATION DECORATOR TESTS")
    print("=" * 80)

    tests = [
        test_basic_validation,
        test_validation_with_tolerance,
        test_validation_failure,
        test_input_output_mapping,
        test_custom_metrics,
        test_method_validation,
        test_enable_disable,
        test_performance_metrics,
        test_multiple_validations,
        test_cosine_similarity,
        test_reference_failure_handling,
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
            failed += 1

    print("\n" + "=" * 80)
    print(f"TESTS COMPLETE: {passed} passed, {failed} failed")
    print("=" * 80)

    # Print sample report
    if get_validation_registry().results:
        print("\nSample validation report:")
        get_validation_registry().print_report()

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
