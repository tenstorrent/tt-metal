# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python modules package for ttml."""

import sys

# Import C++ bindings
from .. import _ttml
from .._recursive_import import _recursive_import_from_ttml

if hasattr(_ttml, "modules"):
    _recursive_import_from_ttml(_ttml.modules, sys.modules[__name__])

from .._ttml.modules import RunMode, ModuleBase

from .module_base import AbstractModuleBase, ModuleDict, ModuleList
from .parameter import Buffer, Parameter

__all__ = [
    "AbstractModuleBase",
    "ModuleBase",
    "ModuleDict",
    "ModuleList",
    "Parameter",
    "Buffer",
    "RunMode",
]
