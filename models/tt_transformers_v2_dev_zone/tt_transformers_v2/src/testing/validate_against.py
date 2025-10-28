# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN Validation Framework

A decorator-based validation system for comparing TTNN implementations against
reference implementations (in PyTorch). Supports automatic input/output
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

import ttnn

from .auto_compose import to_torch_auto_compose
from .metrics import DEFAULT_METRICS, compute_cosine_similarity, compute_max_abs_error, compute_mean_abs_error

# ============================================================================
# Public API
# ============================================================================
#
# Note: Metric functions (compute_max_abs_error, compute_mean_abs_error,
# compute_cosine_similarity) are imported from metrics module and re-exported
# here for convenience.

__all__ = [
    "device_validate_against",
    "host_validate_against",
    "get_validation_registry",
    "enable_validation",
    "clear_validation_results",
    "ValidationResult",
    "ValidationRegistry",
    "compute_max_abs_error",
    "compute_mean_abs_error",
    "compute_cosine_similarity",
    "DEFAULT_METRICS",
]


def get_validation_registry() -> "ValidationRegistry":
    """Get the global validation registry"""
    return _validation_registry


def enable_validation(enabled: bool = True):
    """Enable or disable validation globally"""
    _validation_registry.enabled = enabled


def clear_validation_results():
    """Clear all validation results"""
    _validation_registry.results.clear()


# Convenience wrappers (public) — keep core impl private


def device_validate_against(
    reference_fn: Callable,
    *,
    metrics: Optional[Dict[str, Callable]] = None,
    tolerances: Optional[Dict[str, float]] = None,
    enabled: bool = True,
):
    """
    Convenience wrapper for TTNN-on-device comparison. This is the most useful when the reference function is a TTNN-native function.

    - Assumes the reference function has the same signature as the implementation
      and returns `ttnn.Tensor` (match_signature=True)
    - No input/output mapping; metrics run on-device
    """

    return __validate_against(
        reference_fn=reference_fn,
        input_map=None,
        output_map=None,
        metrics=metrics,
        tolerances=tolerances,
        enabled=enabled,
    )


def host_validate_against(
    reference_fn: Callable,
    *,
    input_to_torch: Optional[Callable] = None,
    output_to_torch: Optional[Callable] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    tolerances: Optional[Dict[str, float]] = None,
    enabled: bool = True,
):
    """
    Convenience wrapper for host/CPU comparison using torch.

    - If not provided, inputs/outputs are automatically converted from TTNN to
      PyTorch using topology-aware auto composition (see auto_compose.to_torch_auto_compose).
      For host-sharded tensors, ensure a default device is set via `ttnn.SetDefaultDevice(...)`
      or provide a custom `input_to_torch` that calls `to_torch_auto_compose(x, device=mesh_device)`.
    - `input_to_torch(args, kwargs) -> (ref_args, ref_kwargs)` can be provided to
      override default input mapping to the torch reference function
    - `output_to_torch(output) -> torch.Tensor` can be provided to override default
      impl-output conversion to torch
    - Reference function is expected to return `torch.Tensor`
    """

    # Default converters: recursively convert any TTNN tensors to torch, auto-compose shards.
    # Non-tensor objects are passed through unchanged.

    def _to_torch_auto(x: Any) -> Any:
        if isinstance(x, ttnn.Tensor):
            # Use auto-compose; relies on tensor.device() or a globally-set default device
            return to_torch_auto_compose(x)
        return x

    def _map_structure(obj: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(obj, (list, tuple)):
            mapped = [_map_structure(x, fn) for x in obj]
            return type(obj)(mapped)
        if isinstance(obj, dict):
            return {k: _map_structure(v, fn) for k, v in obj.items()}
        return fn(obj)

    def _default_input_map(*args, **kwargs):
        ref_args = _map_structure(args, _to_torch_auto)
        ref_kwargs = _map_structure(kwargs, _to_torch_auto)
        return ref_args, ref_kwargs

    def _default_output_map(output):
        return _map_structure(output, _to_torch_auto)

    return __validate_against(
        reference_fn=reference_fn,
        input_map=input_to_torch or _default_input_map,
        output_map=output_to_torch or _default_output_map,
        metrics=metrics,
        tolerances=tolerances,
        enabled=enabled,
    )


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class MetricResult:
    """Per-metric validation outcome"""

    value: float = float("inf")
    passed: bool = False
    error: str = ""


@dataclass
class ValidationResult:
    """Results from a single validation run"""

    function_name: str
    passed: bool
    # Map of metric name to its result (value/pass/fail/error)
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
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
            "avg_speedup": (
                sum(r.execution_time_ref / r.execution_time_impl for r in self.results if r.execution_time_impl > 0)
                / len(self.results)
                if self.results
                else 0.0
            ),
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
                for metric_name, mres in result.metrics.items():
                    if mres.value is not None:
                        try:
                            val_str = f"{mres.value:.6f}"
                        except Exception:
                            val_str = str(mres.value)
                    else:
                        val_str = "-"
                    status = "PASS" if mres.passed else "FAIL"
                    print(f"    {metric_name}: {val_str} — {status}")
                    if mres.error:
                        print(f"      error: {mres.error}")

            # All errors are reported via per-metric entries
            print()
        print("=" * 80 + "\n")


# Global validation registry
_validation_registry = ValidationRegistry()


# ============================================================================
# Validation Decorator
# ============================================================================


# todo)) refactor metrics and tolerances to group them by metrics type -- higher_is_better_metrics = {"pcc", "cosine_similarity"} --> clean up groupings


# todo)) stretch goals:
# - generate unit test automatically from the failed validations
def __validate_against(
    reference_fn: Callable,
    *,
    input_map: Optional[Callable] = None,
    output_map: Optional[Callable] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    tolerances: Optional[Dict[str, float]] = None,
    enabled: bool = True,
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

            # Map inputs for reference function: prefer input_map, else pass-through
            if input_map:
                mapped = input_map(*args, **kwargs)
                # Normalize mapper output:
                # - If (ref_args, ref_kwargs) with kwargs as dict, use directly
                # - Otherwise, treat return as positional args and use empty kwargs
                if isinstance(mapped, tuple) and len(mapped) == 2 and isinstance(mapped[1], dict):
                    ref_args, ref_kwargs = mapped
                else:
                    ref_args = mapped if isinstance(mapped, (list, tuple)) else (mapped,)
                    ref_kwargs = {}
            else:
                ref_args, ref_kwargs = args, kwargs

            # Execute reference
            try:
                start_time = time.perf_counter()
                ref_output = reference_fn(*ref_args, **ref_kwargs)
                ref_time = time.perf_counter() - start_time
            except Exception as e:
                # If reference fails, just return impl output and log error via metrics
                result = ValidationResult(
                    function_name=f"{func.__module__}.{func.__qualname__}",
                    passed=False,
                    metrics={
                        "reference_execution": MetricResult(
                            value=None, passed=False, error=f"Reference execution failed: {str(e)}"
                        )
                    },
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
                    metrics={
                        "output_mapping": MetricResult(
                            value=None, passed=False, error=f"Output mapping failed: {str(e)}"
                        )
                    },
                    execution_time_impl=impl_time,
                    execution_time_ref=ref_time,
                )
                _validation_registry.add_result(result)
                return impl_output

            # Compute metrics
            computed_metrics: Dict[str, MetricResult] = {}
            passed = True

            # Metrics where higher is better (correlation-like metrics)
            higher_is_better_metrics = {"pcc", "cosine_similarity"}

            for metric_name, metric_fn in metrics_to_use.items():
                try:
                    # Only compute metrics we have tolerances for
                    if metric_name in tolerances:
                        value = metric_fn(impl_comparable, ref_comparable)
                        threshold = tolerances[metric_name]

                        if metric_name in higher_is_better_metrics:
                            # For correlation metrics: value should be >= threshold
                            ok = value >= threshold
                            if not ok:
                                passed = False
                                msg = f"{metric_name}={value:.6e} below threshold {threshold:.6e}"
                            computed_metrics[metric_name] = MetricResult(
                                value=value, passed=ok, error=(msg if not ok else None)
                            )
                        else:
                            # For error metrics: value should be <= threshold
                            ok = value <= threshold
                            if not ok:
                                passed = False
                                msg = f"{metric_name}={value:.6e} exceeds tolerance {threshold:.6e}"
                            computed_metrics[metric_name] = MetricResult(
                                value=value, passed=ok, error=(msg if not ok else None)
                            )
                except Exception as e:
                    msg = f"Metric {metric_name} failed: {str(e)}"
                    computed_metrics[metric_name] = MetricResult(value=None, passed=False, error=msg)
                    passed = False

            # Record results
            result = ValidationResult(
                function_name=f"{func.__module__}.{func.__qualname__}",
                passed=passed,
                metrics=computed_metrics,
                execution_time_impl=impl_time,
                execution_time_ref=ref_time,
            )
            _validation_registry.add_result(result)

            return impl_output

        return wrapper

    return decorator
