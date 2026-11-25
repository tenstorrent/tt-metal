# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Process-global TTNN default-device compatibility ownership.

Standalone production passes mesh devices explicitly. This module contains the
only allowed compatibility access to TTNN's process-global default and the only
allowed mutation scope; every scope restores the exact previous object.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DefaultDeviceFallback:
    owner: str
    rationale: str


DEFAULT_DEVICE_FALLBACK_LEDGER = (
    DefaultDeviceFallback(
        "modules.attention._attention_mesh_device",
        "simple/power config may omit mesh_device; wqkv.device is preferred first",
    ),
    DefaultDeviceFallback(
        "modules.embedding._resolve_embedding1d_config",
        "legacy simple constructor permits mesh_device=None after weight-device inference",
    ),
    DefaultDeviceFallback(
        "modules.lm_head._derive_lm_head_mesh_device",
        "legacy simple constructor permits mesh_device=None after weight-device inference",
    ),
    DefaultDeviceFallback(
        "modules.mlp._resolve_mlp1d_mesh",
        "legacy simple constructor permits mesh_device=None after w1.device inference",
    ),
    DefaultDeviceFallback(
        "modules.mlp._resolve_mlp2d_config",
        "legacy 2D constructor permits mesh_device=None after weight-device inference",
    ),
    DefaultDeviceFallback(
        "modules.rmsnorm._derive_rmsnorm_mesh_device",
        "legacy 1D constructor permits mesh_device=None after weight-device inference",
    ),
    DefaultDeviceFallback(
        "modules.rmsnorm._resolve_2d_config",
        "legacy 2D constructor permits mesh_device=None after weight-device inference",
    ),
    DefaultDeviceFallback(
        "modules.rope._resolve_rope_config",
        "legacy simple constructor may omit device after LazyWeight inference",
    ),
    DefaultDeviceFallback(
        "modules.sampling._resolve_penalties1d_config",
        "legacy convenience constructor accepts mesh_device=None",
    ),
    DefaultDeviceFallback(
        "modules.sampling._resolve_sampling1d_config",
        "legacy convenience constructor accepts mesh_device=None",
    ),
)
_FALLBACK_OWNERS = frozenset(record.owner for record in DEFAULT_DEVICE_FALLBACK_LEDGER)
_DEFAULT_DEVICE_LOCK = threading.RLock()
_LOCAL = threading.local()


def _owner_stack() -> list[str]:
    stack = getattr(_LOCAL, "owner_stack", None)
    if stack is None:
        stack = []
        _LOCAL.owner_stack = stack
    return stack


def active_default_device_owners() -> tuple[str, ...]:
    """Return the current thread's nested ownership stack for diagnostics."""

    return tuple(_owner_stack())


def compatibility_default_device(ttnn_api: Any, *, owner: str) -> Any:
    """Read the audited compatibility fallback or raise with an explicit fix."""

    if owner not in _FALLBACK_OWNERS:
        raise ValueError(f"unregistered TTNN default-device fallback owner: {owner}")
    device = ttnn_api.GetDefaultDevice()
    if device is None:
        raise ValueError(f"{owner} requires an explicit mesh_device; no scoped TTNN compatibility default is active")
    return device


@contextmanager
def default_device_scope(ttnn_api: Any, device: Any, *, owner: str) -> Iterator[Any]:
    """Own TTNN's process-global default and restore the exact prior object.

    A re-entrant process lock serializes different threads and supports nested
    same-thread construction. The lock is held for the complete scope, so a
    second owner cannot observe or replace the temporary default.
    """

    if device is None:
        raise ValueError("default_device_scope requires a non-None device")
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("default_device_scope requires a non-empty owner")
    _DEFAULT_DEVICE_LOCK.acquire()
    stack = _owner_stack()
    try:
        previous = ttnn_api.GetDefaultDevice()
        stack.append(owner)
        try:
            try:
                ttnn_api.SetDefaultDevice(device)
            except BaseException:
                # A binding may mutate before raising; make a best effort to
                # put the exact captured object back before exposing failure.
                ttnn_api.SetDefaultDevice(previous)
                raise
            try:
                yield device
            finally:
                ttnn_api.SetDefaultDevice(previous)
        finally:
            popped = stack.pop()
            if popped != owner:
                raise RuntimeError(f"TTNN default-device ownership stack corruption: {popped!r} != {owner!r}")
    finally:
        _DEFAULT_DEVICE_LOCK.release()
