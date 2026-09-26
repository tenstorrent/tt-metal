# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Counts host round-trips inside a model's forward pass (agent rule: no host work in the forward path).

``with HostTransfers() as h: model.layer(...)`` counts calls to the ttnn entry points that move a tensor between host
and device or wait for the device: from_torch, to_torch, synchronize_device and the explicit host/device copies. The
ladder and the profile count only warm calls (after the first chunk), so constants a module builds once and caches do
not count; anything rebuilt per chunk (RoPE tables, masks, page tables uploaded from the host) does.
"""

from __future__ import annotations

from collections import Counter

NAMES = ("from_torch", "to_torch", "synchronize_device", "copy_host_to_device_tensor", "copy_device_to_host_tensor")


class HostTransfers:
    def __init__(self, ttnn_module=None):
        if ttnn_module is None:
            try:
                import ttnn as ttnn_module
            except ImportError:  # a CPU-only environment (selftests): nothing to count
                ttnn_module = None
        self.ttnn, self.calls, self._orig = ttnn_module, Counter(), {}

    def __enter__(self):
        for name in NAMES if self.ttnn is not None else ():
            fn = getattr(self.ttnn, name, None)
            if fn is None:
                continue
            self._orig[name] = fn

            def wrapped(*a, _fn=fn, _name=name, **k):
                self.calls[_name] += 1
                return _fn(*a, **k)

            setattr(self.ttnn, name, wrapped)
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(self.ttnn, name, fn)
        self._orig = {}
        return False

    @property
    def total(self) -> int:
        return sum(self.calls.values())
