# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python models package for ttml.

This package provides Python implementations of models using ttml operations.
"""

import sys

from .. import _ttml

# --- C++ enums and classes from _ttml.models ---
from .._ttml.models import (
    BaseTransformer,
    KvCache,
    KvCacheConfig,
    RunnerType,
    WeightTyingType,
    memory_efficient_runner,
)

# --- C++ submodules without Python package counterparts ---
# Alias _ttml.models.* nanobind submodules so attribute access like
# ttml.models.gpt2.create_gpt2_model(...) works.
gpt2 = _ttml.models.gpt2
sys.modules[f"{__name__}.gpt2"] = gpt2

mlp = _ttml.models.mlp
sys.modules[f"{__name__}.mlp"] = mlp

distributed = _ttml.models.distributed
sys.modules[f"{__name__}.distributed"] = distributed

# --- Python implementations ---
from .linear_regression import LinearRegression, create_linear_regression_model
from .nanogpt import NanoGPT, NanoGPTConfig, create_nanogpt
from .llama import Llama, LlamaConfig

__all__ = [
    # C++ enums / classes
    "BaseTransformer",
    "KvCache",
    "KvCacheConfig",
    "RunnerType",
    "WeightTyingType",
    "memory_efficient_runner",
    # C++ submodules
    "distributed",
    "gpt2",
    "mlp",
    # Python implementations
    "Llama",
    "LlamaConfig",
    "LinearRegression",
    "NanoGPT",
    "NanoGPTConfig",
    "create_linear_regression_model",
    "create_nanogpt",
]
