# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python modules package for ttml.

This package provides Python implementations of module interfaces that mirror
the C++ ModuleBase interface from ttml.
"""

# import C++ nanobind modules
from .._ttml.modules import *

from .module_base import AbstractModuleBase
from .exceptions import (
    ModuleError,
    DuplicateNameError,
    NameNotFoundError,
    UninitializedModuleError,
)
from .parameter import Buffer, Parameter

__all__ = [
    "AbstractModuleBase",
    "Parameter",
    "Buffer",
    "ModuleError",
    "DuplicateNameError",
    "NameNotFoundError",
    "UninitializedModuleError",
]
