# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python modules package for ttml.

This package combines C++ module implementations from _ttml with Python
implementations. C++ classes are imported directly from _ttml.modules.
"""

# Import C++ types
# Note: _ttml is a top-level module, not a subpackage of ttml
import _ttml

RunMode = _ttml.modules.RunMode
ModuleBase = _ttml.modules.ModuleBase
LinearLayer = _ttml.modules.LinearLayer
InferenceMode = _ttml.modules.InferenceMode

# Import Python implementations
from .module_base import AbstractModuleBase, ModuleDict, ModuleList
from .parameter import Buffer, Parameter

__all__ = [
    # C++ classes
    "ModuleBase",
    "RunMode",
    "LinearLayer",
    "InferenceMode",
    # Python classes
    "AbstractModuleBase",
    "ModuleDict",
    "ModuleList",
    "Parameter",
    "Buffer",
]
