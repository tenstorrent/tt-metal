# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python models package for ttml.

This package provides Python implementations of models using ttml operations.
C++ types are imported from _ttml.models and must be available before Python
submodules that depend on them.
"""

# Import C++ types first (needed by Python submodules)
# Note: _ttml is a top-level module, not a subpackage of ttml
import _ttml

RunnerType = _ttml.models.RunnerType
WeightTyingType = _ttml.models.WeightTyingType
memory_efficient_runner = _ttml.models.memory_efficient_runner

# Import Python implementations (can now use C++ types)
from .linear_regression import LinearRegression, create_linear_regression_model
from .nanogpt import NanoGPT, NanoGPTConfig, create_nanogpt

__all__ = [
    # C++ types
    "RunnerType",
    "WeightTyingType",
    "memory_efficient_runner",
    # Python implementations
    "LinearRegression",
    "create_linear_regression_model",
    "NanoGPT",
    "NanoGPTConfig",
    "create_nanogpt",
]
