# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Trace capture helpers.

During ttnn.begin_trace_capture / end_trace_capture, device-side deallocations
(ttnn.deallocate and tensor.deallocate) are treated as writes and trigger
TT_FATAL: Writes are not supported during trace capture.

Call trace_capture_run_begin() before begin_trace_capture and
trace_capture_run_end(token) after end_trace_capture so deallocations in the
captured forward are skipped. Normal inference (and warmup before capture) is
unchanged.
"""

from __future__ import annotations

import contextvars
from typing import Any

import ttnn

_suppress_dealloc: contextvars.ContextVar[bool] = contextvars.ContextVar("molmo2_suppress_dealloc", default=False)

_orig_ttnn_deallocate: Any = None


def _install_deallocate_guard() -> None:
    """Replace ttnn.deallocate once with a wrapper that respects suppression."""
    global _orig_ttnn_deallocate
    if _orig_ttnn_deallocate is not None:
        return

    _orig_ttnn_deallocate = ttnn.deallocate

    def _guarded_deallocate(tensor) -> None:
        if _suppress_dealloc.get():
            return
        return _orig_ttnn_deallocate(tensor)

    ttnn.deallocate = _guarded_deallocate  # type: ignore[assignment]


def trace_capture_run_begin():
    """
    Begin a region where deallocations must be suppressed (e.g. vision/text trace capture).

    Returns:
        Token to pass to trace_capture_run_end.
    """
    _install_deallocate_guard()
    return _suppress_dealloc.set(True)


def trace_capture_run_end(token) -> None:
    """End suppression region started by trace_capture_run_begin."""
    _suppress_dealloc.reset(token)


def dealloc_suppressed() -> bool:
    """True while inside trace_capture_run_begin/end (trace capture forward)."""
    return _suppress_dealloc.get()


def safe_tensor_deallocate(tensor, force: bool = True) -> None:
    """For tensor.deallocate(force); no-op while dealloc is suppressed."""
    if dealloc_suppressed():
        return
    tensor.deallocate(force)
