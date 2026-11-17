# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One captured trace of a forward, replayed on command queue 0."""

from __future__ import annotations

import ttnn

# DRAM to set aside (`trace_region_size` when opening the mesh) for the captured commands of one forward.
TRACE_REGION_SIZE = 100_000_000


class Trace:
    """One captured trace of `forward()`, replayed on command queue 0.

    `forward` must read only inputs that keep their address (tensors kept alive, or `DeviceInput`s),
    and must have run once untraced first: that compiles its programs and allocates the ring attention
    semaphores and buffers, which cannot happen inside a capture. Every replay writes `output`. Use it
    as a context manager, or call `release`, so the trace is released when a check fails.
    """

    def __init__(self, mesh_device, forward) -> None:
        self.mesh_device = mesh_device
        self.trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            self.output = forward()
        except BaseException:
            ttnn.end_trace_capture(mesh_device, self.trace_id, cq_id=0)
            self.release()
            raise
        ttnn.end_trace_capture(mesh_device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        """Release the trace and drop `output`. A second call does nothing."""
        trace_id, self.trace_id = self.trace_id, None
        self.output = None
        if trace_id is not None:
            ttnn.release_trace(self.mesh_device, trace_id)

    def _execute(self) -> None:
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)

    def replay(self) -> ttnn.Tensor:
        """Run once on the current inputs and wait for it; returns `output`."""
        self._execute()
        ttnn.synchronize_device(self.mesh_device)
        return self.output
