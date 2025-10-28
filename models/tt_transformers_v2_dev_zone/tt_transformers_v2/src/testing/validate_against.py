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
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import ttnn

from .auto_compose import to_torch_auto_compose
from .metrics import DEFAULT_METRICS

# ============================================================================
# Public API
# ============================================================================

# Module exports are defined at the package level in __init__.py


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


# todo)) specialize the metric functions for device and host
def device_validate_against(
    reference_fn: Callable,
    *,
    metric_tolerances: Optional[Dict[Any, Any]] = None,
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
        metric_tolerances=metric_tolerances,
        enabled=enabled,
    )


def host_validate_against(
    reference_fn: Callable,
    *,
    input_to_torch: Optional[Callable] = None,
    output_to_torch: Optional[Callable] = None,
    metric_tolerances: Optional[Dict[Any, Any]] = None,
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
        metric_tolerances=metric_tolerances,
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
    metrics: Dict[Any, MetricResult] = field(default_factory=dict)
    execution_time_impl: float = 0.0
    execution_time_ref: float = 0.0
    timestamp: float = field(default_factory=time.time)
    logs: List[str] = field(default_factory=list)


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

    def print_report(self, verbose: bool = False):
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
                    # Use enum value for readability if metric is an Enum
                    name_str = metric_name.value if hasattr(metric_name, "value") else str(metric_name)
                    if mres.value is not None:
                        try:
                            val_str = f"{mres.value:.6f}"
                        except Exception:
                            val_str = str(mres.value)
                    else:
                        val_str = "-"
                    status = "PASS" if mres.passed else "FAIL"
                    print(f"    {name_str}: {val_str} — {status}")
                    if mres.error:
                        print(f"      error: {mres.error}")

            # Print any collected logs for this validation
            if result.logs and verbose:
                print("  Logs:")
                for entry in result.logs:
                    try:
                        msg = str(entry)
                    except Exception:
                        msg = "<unprintable log entry>"
                    print(f"    {msg}")

            # All errors are reported via per-metric entries
            print()
        print("=" * 80 + "\n")


# Global validation registry
_validation_registry = ValidationRegistry()


# ============================================================================
# Validation Decorator
# ============================================================================


class Metric(str, Enum):
    """Enumeration of supported metric names, values match current string keys."""

    MAX_ABS_ERROR = "max_abs_error"
    MEAN_ABS_ERROR = "mean_abs_error"
    PCC = "pcc"


@dataclass
class MetricSpec:
    """Metric specification: name, tolerance, direction, and compute function."""

    tolerance: float
    higher_is_better: bool
    compute_function: Callable[[Any, Any], float]
    name: str = field(default="")


# Registry of built-in metrics with defaults. Tolerances here are sensible
# defaults; callers can override per-validation via `tolerances`.
METRIC_SPECS: Dict[Metric, MetricSpec] = {
    Metric.MAX_ABS_ERROR: MetricSpec(
        name=Metric.MAX_ABS_ERROR.value,
        tolerance=0.0,
        higher_is_better=False,
        compute_function=DEFAULT_METRICS[Metric.MAX_ABS_ERROR.value],
    ),
    Metric.MEAN_ABS_ERROR: MetricSpec(
        name=Metric.MEAN_ABS_ERROR.value,
        tolerance=0.0,
        higher_is_better=False,
        compute_function=DEFAULT_METRICS[Metric.MEAN_ABS_ERROR.value],
    ),
    Metric.PCC: MetricSpec(
        name=Metric.PCC.value,
        tolerance=0.0,
        higher_is_better=True,
        compute_function=DEFAULT_METRICS[Metric.PCC.value],
    ),
}

# Convenience groupings for quick checks
HIGHER_IS_BETTER_METRICS = {m.value for m, spec in METRIC_SPECS.items() if spec.higher_is_better}
LOWER_IS_BETTER_METRICS = {m.value for m, spec in METRIC_SPECS.items() if not spec.higher_is_better}


# Helper: prefer Metric enum as dict key when possible
def _metric_key(key: Any) -> Any:
    try:
        return Metric(key)
    except Exception:
        return key


# Helper: Build active metrics map (name -> compute fn). Accept Metric enum keys for tolerances.
def _normalize_key(k: Any) -> str:
    try:
        # Enum or similar objects with .value as canonical string
        return k.value if hasattr(k, "value") else str(k)
    except Exception:
        return str(k)


# Helper: Prepare metrics, tolerances, and directionality
def _prepare_metric_config(metric_tolerances_input):
    metrics_map = {name: fn for name, fn in DEFAULT_METRICS.items()}
    hib = set(HIGHER_IS_BETTER_METRICS)
    logs_local: List[str] = []
    tol_map: Dict[str, float] = {}

    if not isinstance(metric_tolerances_input, dict):
        logs_local.append(f"metric_tolerances_input must be a dict, got {type(metric_tolerances_input)}")
        metric_tolerances_input = dict()

    if not metric_tolerances_input:
        logs_local.append("no metric tolerances provided")
        metric_tolerances_input = dict()

    for raw_key, spec in metric_tolerances_input.items():
        name = _normalize_key(raw_key)
        if isinstance(spec, MetricSpec):
            tol_map[name] = float(spec.tolerance)
            metrics_map[name] = spec.compute_function
            spec.name = name if spec.name == "" else spec.name
            if spec.higher_is_better:
                hib.add(name)
            else:
                hib.discard(name)
            continue
        try:
            tol_map[name] = float(spec)
        except Exception:
            logs_local.append(f"unrecognized tolerance: {raw_key}: {spec}")

    return metrics_map, hib, tol_map, logs_local


# todo)) stretch goals:
# - generate unit test automatically from the failed validations
def __validate_against(
    reference_fn: Callable,
    *,
    input_map: Optional[Callable] = None,
    output_map: Optional[Callable] = None,
    metric_tolerances: Optional[Dict[Any, Any]] = None,
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
        metric_tolerances: Dictionary specifying tolerances and optionally custom metrics.
            Accepts the following per metric key (str or Metric):
              - float: tolerance only (uses built-in compute + direction)
              - MetricSpec instance
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

    metrics_to_use, higher_is_better_effective, tolerances_map, pre_logs = _prepare_metric_config(metric_tolerances)

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
            logs: List[str] = pre_logs.copy()

            # Map inputs for reference function: prefer input_map, else pass-through
            if input_map:
                _nm = getattr(input_map, "__name__", None) or type(input_map).__name__
                logs.append(f"input_map={_nm}")
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
                logs.append("input_map=pass-through")
                ref_args, ref_kwargs = args, kwargs

            # Execute reference
            try:
                start_time = time.perf_counter()
                ref_output = reference_fn(*ref_args, **ref_kwargs)
                ref_time = time.perf_counter() - start_time
            except Exception as e:
                # If reference fails, just return impl output and log error via metrics
                logs.append(f"reference_execution_error={str(e)}")
                # Record elapsed time until failure
                ref_time = time.perf_counter() - start_time
                result = ValidationResult(
                    function_name=f"{func.__module__}.{func.__qualname__}",
                    passed=False,
                    metrics={
                        "reference_execution": MetricResult(
                            value=None, passed=False, error=f"Reference execution failed: {str(e)}"
                        )
                    },
                    execution_time_impl=impl_time,
                    execution_time_ref=ref_time,
                    logs=logs,
                )
                _validation_registry.add_result(result)
                return impl_output

            # Map outputs for comparison
            # Note: output_map only applies to impl_output to convert it to match ref_output's type
            try:
                _nm = getattr(output_map, "__name__", None) or type(output_map).__name__
                logs.append(f"output_map={_nm}")
                impl_comparable = output_map(impl_output) if output_map else impl_output
                ref_comparable = ref_output  # Reference output is always used as-is
            except Exception as e:
                logs.append(f"output_mapping_error={str(e)}")
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
                    logs=logs,
                )
                _validation_registry.add_result(result)
                return impl_output

            # Compute metrics
            computed_metrics: Dict[Any, MetricResult] = {}
            passed = True

            for metric_name, threshold in tolerances_map.items():
                try:
                    metric_fn = metrics_to_use.get(metric_name)

                    # Store results keyed by enum when available
                    metric_key = _metric_key(metric_name)
                    # If metric function isn't known, record an error
                    if metric_fn is None:
                        computed_metrics[metric_key] = MetricResult(
                            value=None, passed=False, error=f"Unknown metric: {metric_name}"
                        )
                        passed = False
                        continue

                    value = metric_fn(impl_comparable, ref_comparable)

                    # Determine direction using registry when available
                    if metric_name in higher_is_better_effective:
                        ok = value >= threshold
                        err = None
                        if not ok:
                            passed = False
                            err = f"{metric_name}={value:.6e} below threshold {threshold:.6e}"
                        computed_metrics[metric_key] = MetricResult(value=value, passed=ok, error=err)
                    else:
                        ok = value <= threshold
                        err = None
                        if not ok:
                            passed = False
                            err = f"{metric_name}={value:.6e} exceeds tolerance {threshold:.6e}"
                        computed_metrics[metric_key] = MetricResult(value=value, passed=ok, error=err)
                except Exception as e:
                    msg = f"Metric {metric_name} failed: {str(e)}"
                    computed_metrics[metric_key] = MetricResult(value=None, passed=False, error=msg)
                    passed = False

            # Record results
            try:
                pass_count = sum(1 for v in computed_metrics.values() if v.passed)
                fail_count = sum(1 for v in computed_metrics.values() if not v.passed)
                logs.append(f"metrics={pass_count}_pass,{fail_count}_fail")
            except Exception:
                pass
            result = ValidationResult(
                function_name=f"{func.__module__}.{func.__qualname__}",
                passed=passed,
                metrics=computed_metrics,
                execution_time_impl=impl_time,
                execution_time_ref=ref_time,
                logs=logs,
            )
            _validation_registry.add_result(result)

            return impl_output

        return wrapper

    return decorator
