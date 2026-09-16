# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ttpoly.groundtruth — the single high-precision ground-truth provider.

Consolidation map:
    ``ground_truth.py`` + ``compute_mpmath_300b_ulp`` -> ``groundtruth/reference.py``

This package owns the *one* golden-value provider for the library. It computes
the high-precision reference with **mpmath at 300-bit precision** as the PRIMARY
path for every activation whose mathematics is expressible in mpmath, and falls
back to FP64 (via the legacy ``ground_truth`` module) for the piecewise-linear /
clamp activations that mpmath buys nothing for.

Public API (see ``reference.py`` for details)::

    reference_value(activation, x)      -> np.ndarray (float64) golden values
    reference_source(activation)        -> "mpmath_300b" | "fp64_fallback"
    reference_sources()                 -> {activation: source} for all activations
"""

from .reference import (
    DEFAULT_MPMATH_PREC_BITS,
    ReferenceCache,
    reference_source,
    reference_sources,
    reference_value,
)

# The relocated golden-activation provider (byte-identical port of the legacy
# ``ground_truth`` module). These are the FP64 / metadata helpers the rest of
# ttpoly needs (get_activation, domains, sollya exprs, torch-native fns,
# piecewise / asymptotic config) so nothing has to import the root module.
from .activations import (
    compute_ground_truth,
    get_activation,
    get_activation_domain,
    get_activation_function,
    get_all_activations,
    get_asymptotic_config_for_piece,
    get_critical_points_from_config,
    get_piece_index_for_segment,
    get_piecewise_breakpoints_from_config,
    get_sollya_expression,
    get_sollya_expression_for_piece,
    get_torch_native_function,
    is_piecewise_linear_activation,
    is_piecewise_smooth_activation,
    load_activation_config,
    supports_sollya,
)

__all__ = [
    "reference_value",
    "reference_source",
    "reference_sources",
    "ReferenceCache",
    "DEFAULT_MPMATH_PREC_BITS",
    # relocated ground-truth provider
    "get_activation",
    "get_activation_function",
    "get_activation_domain",
    "get_all_activations",
    "get_sollya_expression",
    "get_sollya_expression_for_piece",
    "get_torch_native_function",
    "get_asymptotic_config_for_piece",
    "get_piece_index_for_segment",
    "get_piecewise_breakpoints_from_config",
    "get_critical_points_from_config",
    "load_activation_config",
    "supports_sollya",
    "is_piecewise_linear_activation",
    "is_piecewise_smooth_activation",
    "compute_ground_truth",
]
