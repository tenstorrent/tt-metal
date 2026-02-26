# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTML Python package.

This package provides Python bindings and implementations for the TTML
(Tenstorrent Machine Learning) library. All symbols from the _ttml C++
extension are automatically imported here, with Python implementations
taking precedence when they exist.
"""

import sys
import ttnn

# Try to import _ttml from the build directory first (when using .pth file with
# build_metal.sh --build-tt-train), then fall back to local package (standalone pip install)
try:
    import _ttml

    # Ensure _ttml is also visible as a submodule of this package for relative imports
    sys.modules[__name__ + "._ttml"] = _ttml

except ImportError:
    from . import _ttml

from ._recursive_import import _recursive_import_from_ttml

# Recursively import all _ttml symbols into this module
# Python implementations in subpackages will take precedence
_recursive_import_from_ttml(_ttml, sys.modules[__name__])
