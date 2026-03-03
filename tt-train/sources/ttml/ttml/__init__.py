# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTML Python package.

This package provides Python bindings and implementations for the TTML
(Tenstorrent Machine Learning) library. C++ symbols from the _ttml nanobind
extension are explicitly re-exported here and in subpackage __init__.py files.
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

# --- Top-level symbols from _ttml ---
from ._ttml import NamedParameters

# --- Python subpackages ---
from . import autograd
from . import models
from . import modules

# --- Re-export _ttml submodules that have no Python package counterpart ---
# These are pure C++ nanobind submodules; making them attributes of ttml
# and registering in sys.modules allows both attribute access (ttml.ops.loss.*)
# and import statements (from ttml import ops).
ops = _ttml.ops
sys.modules[f"{__name__}.ops"] = ops

core = _ttml.core
sys.modules[f"{__name__}.core"] = core

optimizers = _ttml.optimizers
sys.modules[f"{__name__}.optimizers"] = optimizers
