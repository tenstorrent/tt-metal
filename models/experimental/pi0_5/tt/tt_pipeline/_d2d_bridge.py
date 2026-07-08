# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native, collapsed D2DBridge for the standalone pi0.5 streamed-denoise port.

Verified: ``D2DBridge`` is only passed to ``Pipeline(bridges)`` for topology bookkeeping;
it is NEVER ``__call__``-ed on the streamed path. So this keeps ONLY: the ctor (two pre-bound
meshes, ``_``-prefixed attrs, ``_device=_mesh_b``, ``_bypass_tensor_wrapping=True``); the
producer/consumer/transport/tag/mesh_a/mesh_b accessors; ``to_device`` / ``set_device_state``
no-ops; ``close()``. DROPPED: call / _call_eager / _call_traced / _capture / _replay /
_flatten / _join / _rewrap_like. ZERO tt_symbiote imports.
"""
from __future__ import annotations

import ttnn  # noqa: F401  (kept for parity; close() may sync in future)

from ._module import StatelessTTNNModule
from ._trace import trace_enabled
from ._transport import SplitSocketTransport

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"


@trace_enabled
class D2DBridge(StatelessTTNNModule):
    """Directed dependency edge (producer -> consumer) across two pre-bound submeshes."""

    def __init__(self, module_a, module_b, *, transport=None, tag=None):
        super().__init__()
        mesh_a = getattr(module_a, "device", None)
        mesh_b = getattr(module_b, "device", None)
        if mesh_a is None or mesh_b is None:
            raise ValueError(
                "D2DBridge: both modules must already be bound to a device before being passed to "
                f"the bridge (module_a.device={mesh_a!r}, module_b.device={mesh_b!r})."
            )
        if mesh_a is mesh_b:
            raise ValueError("D2DBridge: module_a and module_b must be on TWO DIFFERENT devices")
        # underscore attrs: a set_device() walk skips _-prefixed attrs (pre-bound children).
        self._module_a = module_a
        self._module_b = module_b
        self._mesh_a = mesh_a
        self._mesh_b = mesh_b
        self._tag = tag
        self._transport = transport or SplitSocketTransport()
        self._owns_transport = transport is None
        self._bypass_tensor_wrapping = True
        self._device = self._mesh_b  # expose OUTPUT mesh as .device

    def to_device(self, device):
        return self

    def set_device_state(self, device_state=None):
        return self

    @property
    def mesh_a(self):
        return self._mesh_a

    @property
    def mesh_b(self):
        return self._mesh_b

    @property
    def producer(self):
        return self._module_a

    @property
    def consumer(self):
        return self._module_b

    @property
    def transport(self):
        return self._transport

    @property
    def tag(self):
        return self._tag

    def forward(self, *args, **kwds):
        raise RuntimeError("D2DBridge is topology bookkeeping only; forward() should never be invoked.")

    def close(self):
        for m in (self._mesh_a, self._mesh_b):
            if m is not None:
                try:
                    ttnn.synchronize_device(m)
                except Exception:
                    pass
        if self._owns_transport:
            try:
                self._transport.close()
            except Exception:
                pass
