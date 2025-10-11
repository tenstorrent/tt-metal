# TTTv2 Testing Utilities

# Validation framework
# Metrics
from .metrics import (
    DEFAULT_METRICS,
    comp_allclose,
    compute_cosine_similarity,
    compute_max_abs_error,
    compute_mean_abs_error,
    compute_pcc,
)
from .validate_against import (
    ValidationRegistry,
    ValidationResult,
    clear_validation_results,
    enable_validation,
    get_validation_registry,
    validate_against,
)

__all__ = [
    # Validation framework
    "validate_against",
    "get_validation_registry",
    "enable_validation",
    "clear_validation_results",
    "ValidationResult",
    "ValidationRegistry",
    # Metrics
    "compute_max_abs_error",
    "compute_mean_abs_error",
    "compute_pcc",
    "compute_cosine_similarity",
    "comp_allclose",
    "DEFAULT_METRICS",
]
