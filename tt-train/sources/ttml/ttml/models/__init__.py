# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python models package for ttml.

This package provides Python implementations of models using ttml operations.
"""

# Import Python implementations
from .linear_regression import LinearRegression, create_linear_regression_model
from .nanogpt import NanoGPT, NanoGPTConfig, create_nanogpt

__all__ = [
    "LinearRegression",
    "create_linear_regression_model",
    "NanoGPT",
    "NanoGPTConfig",
    "create_nanogpt",
]
