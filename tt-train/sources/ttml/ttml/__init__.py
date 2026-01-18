# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTML Python package.

This package provides Python bindings and implementations for the TTML
(Tenstorrent Machine Learning) library. All symbols from the _ttml C++
extension are automatically imported here, with Python implementations
taking precedence when they exist.
"""


def _initialize():
    """Initialize the module, handling dev vs installed environments."""
    import os
    from pathlib import Path

    def _preload_dev_tt_metal_libraries(tt_metal_home):
        """
        Preload TT Metal libraries from development build directory.
        """
        import ctypes

        lib_dir = tt_metal_home / "build" / "lib"

        if not lib_dir.exists():
            return False

        # Load required libraries
        required_libs = ["libtt_metal.so", "_ttnncpp.so"]

        for filename in required_libs:
            lib_path = lib_dir / filename

            if not lib_path.exists():
                return False

            try:
                ctypes.cdll.LoadLibrary(str(lib_path))
            except OSError:
                return False

        return True

    # Determine TT Metal home directory
    tt_metal_home = os.getenv("TT_METAL_HOME")
    if tt_metal_home:
        tt_metal_home = Path(tt_metal_home)
    else:
        tt_metal_home = Path.home() / "tt-metal"

    # Always try to preload libraries in dev environment
    if tt_metal_home.exists():
        _preload_dev_tt_metal_libraries(tt_metal_home)

    # Add build directory to path for finding _ttml
    ttml_dir = tt_metal_home / "build" / "tt-train" / "sources" / "ttml"

    import sys

    if str(ttml_dir) not in sys.path:
        sys.path.insert(0, str(ttml_dir))

    import importlib.util

    # Check if _ttml can be found in the build directory
    ttml_module = importlib.util.find_spec("_ttml", package=None)
    return ttml_module is not None and ttml_dir.exists()


# Initialize the module
_use_local_ttml = _initialize()

import sys

if _use_local_ttml:
    from _ttml import *  # noqa: F401, F403
    import _ttml
else:
    from ._ttml import *  # noqa: F401, F403
    from . import _ttml

# Import _recursive_import and apply it
from ._recursive_import import _recursive_import_from_ttml

_recursive_import_from_ttml(_ttml, sys.modules[__name__])
