# TTTv2 Testing Utilities

# Validation framework
# Metrics
from .auto_compose import to_torch_auto_compose
from .metrics import (
    DEFAULT_METRICS,
    comp_allclose,
    compute_max_abs_error,
    compute_mean_abs_error,
    compute_pcc_host,
    compute_pcc,
)
from .validate_against import (
    ValidationRegistry,
    ValidationResult,
    Metric,
    clear_validation_results,
    compare_to_ttnn,
    enable_validation,
    get_validation_registry,
    compare_to_torch,
    MetricSpec,
)

__all__ = [
    # Validation framework
    "compare_to_ttnn",
    "compare_to_torch",
    "get_validation_registry",
    "enable_validation",
    "clear_validation_results",
    "ValidationResult",
    "ValidationRegistry",
    "Metric",
    "MetricSpec",
    # Metrics
    "compute_max_abs_error",
    "compute_mean_abs_error",
    "compute_pcc",
    "compute_pcc_host",
    "comp_allclose",
    "DEFAULT_METRICS",
    # Auto compose
    "to_torch_auto_compose",
]
