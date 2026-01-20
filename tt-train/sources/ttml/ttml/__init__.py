# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTML Python package.

This package provides Python bindings and implementations for the TTML
(Tenstorrent Machine Learning) library. All symbols from the _ttml C++
extension are automatically imported here, with Python implementations
taking precedence when they exist.
"""

import sys
from . import _ttml
from ._recursive_import import _recursive_import_from_ttml

# Recursively import all _ttml symbols into this module
# Python implementations in subpackages will take precedence
_recursive_import_from_ttml(_ttml, sys.modules[__name__])
