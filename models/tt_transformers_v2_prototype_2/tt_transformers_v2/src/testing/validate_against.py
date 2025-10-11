#!/usr/bin/env python3
"""
TTNN Validation Framework

A decorator-based validation system for comparing TTNN implementations against
reference implementations (typically PyTorch). Supports automatic input/output
mapping, metric computation, and result collection.

Key Features:
- Automatic comparison of TTNN vs reference implementations
- TTNN-native metric computation (stays on device until final scalar)
- Flexible input/output mapping
- Built-in metrics: max_abs_error, mean_abs_error, cosine_similarity
- Performance tracking
- Result registry for batch reporting
"""

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ValidationResult:
    """Results from a single validation run"""

    function_name: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time_impl: float = 0.0
    execution_time_ref: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ValidationRegistry:
    """Global registry for validation results"""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.enabled = True

    def add_result(self, result: ValidationResult):
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all validations"""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0}

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0.0,
            "avg_speedup": sum(
                r.execution_time_ref / r.execution_time_impl for r in self.results if r.execution_time_impl > 0
            )
            / len(self.results)
            if self.results
            else 0.0,
        }

    def print_report(self):
        """Print detailed validation report"""
        summary = self.get_summary()
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        print(f"Total validations: {summary['total']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"Average speedup: {summary['avg_speedup']:.2f}x")
        print()

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} - {result.function_name}")
            print(
                f"  Execution time: impl={result.execution_time_impl*1000:.2f}ms, ref={result.execution_time_ref*1000:.2f}ms"
            )

            if result.metrics:
                print(f"  Metrics:")
                for metric, value in result.metrics.items():
                    print(f"    {metric}: {value:.6f}")

            if result.errors:
                print(f"  Errors:")
                for error in result.errors:
                    print(f"    - {error}")
            print()
        print("=" * 80 + "\n")


# Global validation registry
_validation_registry = ValidationRegistry()


# Import metric functions
from .metrics import DEFAULT_METRICS, _compute_cosine_similarity, _compute_max_abs_error, _compute_mean_abs_error

# ============================================================================
# Validation Decorator
# ============================================================================


def validate_against(
    reference_fn: Callable,
    input_map: Optional[Callable] = None,
    output_map: Optional[Callable] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    tolerances: Optional[Dict[str, float]] = None,
    enabled: bool = True,
    match_signature: bool = False,
):
    """
    Decorator to validate a function against a reference implementation.

    Args:
        reference_fn: Reference function to compare against
        input_map: Maps decorated function inputs to reference function inputs
                   Signature: (args, kwargs) -> (ref_args, ref_kwargs)
                   If None, inputs are passed as-is
        output_map: Converts impl output to match ref output's type
                    Signature: (output) -> comparable_output
                    Applied ONLY to impl_output to convert it to ref_output's type
                    Common use: lambda x: ttnn.to_torch(x).squeeze() to convert ttnn → torch
                    If None, outputs are used as-is (both must already be same type)
        metrics: Dictionary of metric_name -> metric_function(impl_out, ref_out) -> float
                 Default metrics: max_abs_error, mean_abs_error, cosine_similarity
        tolerances: Dictionary of metric_name -> max_acceptable_value
                    Validation fails if any metric exceeds its tolerance
        enabled: Whether validation is enabled (can disable globally via registry)
        match_signature: If True, reference_fn has the same signature as the decorated
                        function and will be called with identical args/kwargs.
                        This allows using wrapper functions without complex input_map.
                        With TTNN-native metrics, reference should return ttnn.Tensor
                        for on-device computation (100-1000× faster).

    Examples:
        # Pattern 1: TTNN-native metrics (recommended, 100-1000× faster!)
        # Both impl and ref return ttnn.Tensor, no output_map needed
        def _reference_impl(self, x):
            x_torch = ttnn.to_torch(x).squeeze(0)
            result_torch = torch.matmul(x_torch, self.weight_torch)
            # Convert back to TTNN for on-device metrics!
            return ttnn.from_torch(result_torch.unsqueeze(0), device=self.device, ...)

        @validate_against(
            reference_fn=lambda self, x: self._reference_impl(x),
            match_signature=True,
            tolerances={'max_abs_error': 1e-3}
        )
        def __call__(self, x):
            return ttnn.matmul(x, self.weight)

        # Pattern 2: PyTorch metrics (when reference returns torch.Tensor)
        # Use output_map to convert impl output (ttnn.Tensor) to match ref (torch.Tensor)
        @validate_against(
            reference_fn=torch.nn.functional.rms_norm,
            input_map=lambda args, kwargs: (
                (ttnn.to_torch(args[1]).squeeze(),),
                {'eps': args[0].eps}
            ),
            output_map=lambda x: ttnn.to_torch(x).squeeze(),  # Convert impl: ttnn → torch
            tolerances={'max_abs_error': 1e-3}
        )
        def __call__(self, x):
            return ttnn.rms_norm(x, self.weight, self.eps)  # Returns ttnn.Tensor
    """

    # Use default metrics from ttnn_metrics module
    default_metrics = DEFAULT_METRICS.copy()

    if metrics:
        default_metrics.update(metrics)

    metrics_to_use = default_metrics
    tolerances = tolerances or {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if validation is enabled
            if not enabled or not _validation_registry.enabled:
                return func(*args, **kwargs)

            # Execute implementation
            start_time = time.perf_counter()
            impl_output = func(*args, **kwargs)
            impl_time = time.perf_counter() - start_time

            # Map inputs for reference function
            if match_signature:
                # Reference function has same signature, call with same args/kwargs
                ref_args, ref_kwargs = args, kwargs
            elif input_map:
                # Use custom input mapping
                ref_args, ref_kwargs = input_map(args, kwargs)
            else:
                # Pass through as-is
                ref_args, ref_kwargs = args, kwargs

            # Execute reference
            try:
                start_time = time.perf_counter()
                ref_output = reference_fn(*ref_args, **ref_kwargs)
                ref_time = time.perf_counter() - start_time
            except Exception as e:
                # If reference fails, just return impl output and log error
                result = ValidationResult(
                    function_name=f"{func.__module__}.{func.__qualname__}",
                    passed=False,
                    errors=[f"Reference execution failed: {str(e)}"],
                    execution_time_impl=impl_time,
                )
                _validation_registry.add_result(result)
                return impl_output

            # Map outputs for comparison
            # Note: output_map only applies to impl_output to convert it to match ref_output's type
            try:
                impl_comparable = output_map(impl_output) if output_map else impl_output
                ref_comparable = ref_output  # Reference output is always used as-is
            except Exception as e:
                result = ValidationResult(
                    function_name=f"{func.__module__}.{func.__qualname__}",
                    passed=False,
                    errors=[f"Output mapping failed: {str(e)}"],
                    execution_time_impl=impl_time,
                    execution_time_ref=ref_time,
                )
                _validation_registry.add_result(result)
                return impl_output

            # Compute metrics
            computed_metrics = {}
            errors = []
            passed = True

            for metric_name, metric_fn in metrics_to_use.items():
                try:
                    value = metric_fn(impl_comparable, ref_comparable)
                    computed_metrics[metric_name] = value

                    # Check tolerance
                    if metric_name in tolerances:
                        if value > tolerances[metric_name]:
                            passed = False
                            errors.append(f"{metric_name}={value:.6e} exceeds tolerance {tolerances[metric_name]:.6e}")
                except Exception as e:
                    errors.append(f"Metric {metric_name} failed: {str(e)}")
                    passed = False

            # Record results
            result = ValidationResult(
                function_name=f"{func.__module__}.{func.__qualname__}",
                passed=passed,
                metrics=computed_metrics,
                errors=errors,
                execution_time_impl=impl_time,
                execution_time_ref=ref_time,
            )
            _validation_registry.add_result(result)

            return impl_output

        return wrapper

    return decorator


# ============================================================================
# Public API
# ============================================================================
#
# Note: Metric functions (_compute_max_abs_error, _compute_mean_abs_error,
# _compute_cosine_similarity) are imported from metrics module and re-exported
# here for convenience.


def get_validation_registry() -> ValidationRegistry:
    """Get the global validation registry"""
    return _validation_registry


def enable_validation(enabled: bool = True):
    """Enable or disable validation globally"""
    _validation_registry.enabled = enabled


def clear_validation_results():
    """Clear all validation results"""
    _validation_registry.results.clear()


# Re-export metric functions for backward compatibility
__all__ = [
    "validate_against",
    "get_validation_registry",
    "enable_validation",
    "clear_validation_results",
    "ValidationResult",
    "ValidationRegistry",
    "_compute_max_abs_error",
    "_compute_mean_abs_error",
    "_compute_cosine_similarity",
    "DEFAULT_METRICS",
]
