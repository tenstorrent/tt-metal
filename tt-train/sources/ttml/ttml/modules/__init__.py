# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python modules package for ttml.

This package provides Python implementations of module interfaces that mirror
the C++ ModuleBase interface from ttml.
"""

import sys
from .module_base import AbstractModuleBase
from .exceptions import (
    ModuleError,
    DuplicateNameError,
    NameNotFoundError,
    UninitializedModuleError,
)
from .parameter import Buffer, Parameter

# Import Python implementations first so they take precedence
# Then conditionally import from _ttml.modules for any symbols not overridden
from .. import _ttml
from .._recursive_import import _recursive_import_from_ttml

if hasattr(_ttml, "modules"):
    _recursive_import_from_ttml(_ttml.modules, sys.modules[__name__])

__all__ = [
    "AbstractModuleBase",
    "Parameter",
    "Buffer",
    "ModuleError",
    "DuplicateNameError",
    "NameNotFoundError",
    "UninitializedModuleError",
]
