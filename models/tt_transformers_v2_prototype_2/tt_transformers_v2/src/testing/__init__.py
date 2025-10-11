# TTTv2 Testing Utilities

# Validation framework
# Metrics
from .metrics import DEFAULT_METRICS, _compute_cosine_similarity, _compute_max_abs_error, _compute_mean_abs_error
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
    "_compute_max_abs_error",
    "_compute_mean_abs_error",
    "_compute_cosine_similarity",
    "DEFAULT_METRICS",
]
