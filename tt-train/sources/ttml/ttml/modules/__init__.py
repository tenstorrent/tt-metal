# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Python modules package for ttml."""

# C++ bindings from _ttml.modules
# Use _ttml directly (compiled .so has no __path__, so .._ttml fails)
from _ttml.modules import InferenceMode, ModuleBase, RunMode

# Python implementations
from .embedding import Embedding, VocabParallelEmbedding
from .linear import LinearLayer, ColumnParallelLinear, RowParallelLinear
from .lora import LoraConfig, LoraLinear, LoraColumnParallelLinear, LoraRowParallelLinear, LoraModel
from .module_base import AbstractModuleBase, ModuleDict, ModuleList
from .parameter import Buffer, Parameter

__all__ = [
    # C++ bindings
    "InferenceMode",
    "ModuleBase",
    "RunMode",
    # Python classes
    "ColumnParallelLinear",
    "RowParallelLinear",
    "AbstractModuleBase",
    "Buffer",
    "Embedding",
    "LinearLayer",
    "LoraColumnParallelLinear",
    "LoraConfig",
    "LoraLinear",
    "LoraModel",
    "LoraRowParallelLinear",
    "ModuleDict",
    "ModuleList",
    "Parameter",
    "VocabParallelEmbedding",
]
