# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python modules package for ttml."""

# C++ bindings from _ttml.modules
# Use _ttml directly (compiled .so has no __path__, so .._ttml fails)
from _ttml.modules import InferenceMode, ModuleBase, RunMode

# Python implementations
from .linear import LinearLayer
from .lora import LoraConfig, LoraLinear, LoraModel
from .module_base import AbstractModuleBase, ModuleDict, ModuleList, TransformerBase
from .parameter import Buffer, Parameter, TensorMetadata

__all__ = [
    "AbstractModuleBase",
    "Buffer",
    "InferenceMode",
    "LinearLayer",
    "LoraConfig",
    "LoraLinear",
    "LoraModel",
    "ModuleBase",
    "ModuleDict",
    "ModuleList",
    "Parameter",
    "RunMode",
    "TensorMetadata",
    "TransformerBase",
]
