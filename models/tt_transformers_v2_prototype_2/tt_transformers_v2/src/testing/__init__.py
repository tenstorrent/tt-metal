# TTTv2 Testing Utilities

# Validation framework
from .validation import (
    validate_against,
    get_validation_registry,
    enable_validation,
    clear_validation_results,
    ValidationResult,
    ValidationRegistry,
)

# Metrics
from .metrics import (
    _compute_max_abs_error,
    _compute_mean_abs_error,
    _compute_cosine_similarity,
    DEFAULT_METRICS,
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
