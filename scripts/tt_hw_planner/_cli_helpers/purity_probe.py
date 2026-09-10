# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime purity probe: assert a stub's forward does not pull tensors off
the device (no ttnn.to_torch / .cpu() / from_torch mid-forward).

Complements the static G1b AST check: if a stub finds a way to trigger a
host-copy that the static check doesn't recognize (dynamic imports, custom
copy helpers, exotic APIs), this runtime probe catches it by observing the
ttnn transport calls directly."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable


@dataclass
class PurityViolation(AssertionError):
    calls: list

    def __str__(self) -> str:
        n = len(self.calls)
        head = "; ".join(self.calls[:5])
        tail = f" (+{n - 5} more)" if n > 5 else ""
        return f"stub forward performed {n} host-copy call(s): {head}{tail}"


@contextlib.contextmanager
def _monitor_ttnn_transport():
    try:
        import ttnn
    except Exception:  # noqa: BLE001
        yield []
        return

    calls: list = []
    orig_to_torch = getattr(ttnn, "to_torch", None)
    orig_from_torch = getattr(ttnn, "from_torch", None)

    def _wrap_to_torch(*args, **kwargs):
        calls.append("ttnn.to_torch(...)")
        return orig_to_torch(*args, **kwargs)

    def _wrap_from_torch(*args, **kwargs):
        calls.append("ttnn.from_torch(...)")
        return orig_from_torch(*args, **kwargs)

    if orig_to_torch is not None:
        ttnn.to_torch = _wrap_to_torch
    if orig_from_torch is not None:
        ttnn.from_torch = _wrap_from_torch
    try:
        yield calls
    finally:
        if orig_to_torch is not None:
            ttnn.to_torch = orig_to_torch
        if orig_from_torch is not None:
            ttnn.from_torch = orig_from_torch


def assert_pure_on_device(fn: Callable, *args, **kwargs):
    """Run fn(*args, **kwargs) with ttnn.to_torch / ttnn.from_torch monitored.
    Raises PurityViolation if either is called during the call. Returns the
    function's own return value on success."""
    with _monitor_ttnn_transport() as calls:
        out = fn(*args, **kwargs)
    if calls:
        raise PurityViolation(calls)
    return out


__all__ = ["assert_pure_on_device", "PurityViolation"]
