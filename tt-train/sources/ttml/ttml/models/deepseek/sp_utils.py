# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sequence-parallel (SP) helpers for DeepSeek modules.

Pure Python module: holds a process-wide SP flag and exposes accessors used
by the deepseek modules. Importing this file has no side effects on running
code paths — until ``set_sp_state(enabled=True, ...)`` is called the helpers
report SP as disabled and callers leave the existing code path untouched.
"""

from __future__ import annotations

from typing import Optional

import ttml


_sp_enabled: bool = False
_sp_tp_axis: Optional[int] = None


def set_sp_state(*, enabled: bool, tp_axis_name: str = "tp") -> None:
    """Configure SP for the current process.

    Called from the training entry point once the mesh is open. ``tp_axis_name``
    must resolve to a real mesh axis with size > 1 when ``enabled=True``.
    """
    global _sp_enabled, _sp_tp_axis
    if not enabled:
        _sp_enabled = False
        _sp_tp_axis = None
        return
    mesh = ttml.maybe_mesh()
    if mesh is None or not mesh.has_axis(tp_axis_name) or mesh.axis_size(tp_axis_name) <= 1:
        raise ValueError(f"set_sp_state(enabled=True) requires an open mesh with axis " f"{tp_axis_name!r} of size > 1")
    _sp_enabled = True
    _sp_tp_axis = mesh.axis_index(tp_axis_name)


def sp_enabled() -> bool:
    """True iff sequence parallelism is currently active."""
    return _sp_enabled


def sp_tp_axis() -> Optional[int]:
    """TP cluster axis index when SP is active, else None."""
    return _sp_tp_axis
