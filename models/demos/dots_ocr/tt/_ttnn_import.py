# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Safe TTNN import for dots_ocr.

Some environments ship a partial or version-mismatched `ttnn` Python package that
raises AttributeError during import (e.g. missing symbols on `ttnn._ttnn.device`).
We catch **all** import-time failures so CPU-only tests and demos can still import
reference code without a working TTNN runtime.
"""

from __future__ import annotations

from typing import Any, Optional

_ttnn_module: Optional[Any] = None
_import_error: Optional[BaseException] = None


def get_ttnn():
    """
    Return the `ttnn` module, or None if import failed.

    Caches success/failure so repeated calls do not re-trigger broken imports.
    """
    global _ttnn_module, _import_error
    if _ttnn_module is not None:
        return _ttnn_module
    if _import_error is not None:
        return None
    try:
        import ttnn as t  # noqa: WPS433 — runtime import by design

        _ttnn_module = t
        return _ttnn_module
    except BaseException as e:  # noqa: BLE001 — intentional: catch broken ttnn installs
        _import_error = e
        return None


def ttnn_import_error() -> Optional[BaseException]:
    """Return the exception from the last failed import attempt, if any."""
    get_ttnn()  # ensure attempted
    return _import_error


def ttnn_available() -> bool:
    return get_ttnn() is not None
