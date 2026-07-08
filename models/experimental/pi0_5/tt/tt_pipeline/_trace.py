# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native, collapsed trace registry for the standalone pi0.5 streamed-denoise port.

This is the ~30-line distillation of ``tt_symbiote.core.run_config`` that the streamed
driver actually needs. The full run-mode machinery (TracedRun / DispatchManager /
DistributedConfig / _make_cache_key / get_tensor_run_implementation) is dropped: the
streamed driver drives capture/replay explicitly. We keep ONLY:

  * ``_TRACE_RUNNING``     -- the module-global the weight-prep guards read deferred.
  * ``trace_running()``    -- context manager that latches it for capture regions.
  * ``trace_enabled`` / ``trace_disabled`` / ``is_trace_enabled`` -- the registry the
    ``@trace_enabled`` decorations on Pipeline / D2DBridge / stage / expert block use as
    no-ops (they no longer pull anything into a TracedRun lifecycle -- that is gone).

ZERO tt_symbiote imports. Imports with tt_symbiote NOT installed.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Set, Type

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

_TRACE_ENABLED_CLASSES: Set[Type] = set()
_TRACE_DISABLED_CLASSES: Set[Type] = set()
_TRACE_RUNNING = False


def trace_enabled(cls: Type) -> Type:
    """Register a class as trace-enabled (registry no-op for this port)."""
    _TRACE_ENABLED_CLASSES.add(cls)
    return cls


def trace_disabled(cls: Type) -> Type:
    """Register a class as trace-disabled even if its parent is trace-enabled."""
    _TRACE_DISABLED_CLASSES.add(cls)
    return cls


def is_trace_enabled(module) -> bool:
    """True iff module's class is trace-enabled and not trace-disabled."""
    return isinstance(module, tuple(_TRACE_ENABLED_CLASSES)) and not isinstance(module, tuple(_TRACE_DISABLED_CLASSES))


@contextmanager
def trace_running():
    """Latch ``_TRACE_RUNNING`` True for the duration of a trace capture region."""
    global _TRACE_RUNNING
    was = _TRACE_RUNNING
    _TRACE_RUNNING = True
    try:
        yield
    finally:
        _TRACE_RUNNING = was
