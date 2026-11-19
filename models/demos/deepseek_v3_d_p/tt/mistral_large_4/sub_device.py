# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""perf 16: the shared expert overlapped with the routed experts' dispatch on disjoint cores (MOE_OVERLAP).

As tt-metal's DeepSeek prefill (models/demos/deepseek_v3_d_p/tt/moe/tt_moe.py overlap_shared_expert_with_dispatch):
a sub-device manager splits the Tensix grid into a dispatch strip (row 0) and a shared-expert strip (rows 1..);
dispatch runs on the first and the shared expert's matmuls on the second, so Fast Dispatch runs them concurrently.
The manager is loaded around that region only (every other op keeps the whole grid).

A sub-device manager cannot be loaded or cleared inside a trace capture, so a capture that meets one is split there
(as models/demos/deepseek_v3_d_p/utils/sub_device_trace.py): the load / clear runs on the host between two trace
segments, at capture and at every replay. ``begin_capture`` / ``end_capture`` / ``execute`` / ``release`` stand in
for ttnn's trace calls everywhere the package captures; with MOE_OVERLAP off they are those calls (one segment).
"""

from __future__ import annotations

import ttnn

from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

_overlaps = {}  # id(mesh) -> Overlap
_capturing = []  # the Trace being captured (at most one)
last_segments = 1  # segments of the latest finished capture (for the tests' records)


def enabled() -> bool:
    return settings.get("MOE_OVERLAP") == "shared_dispatch"


class Overlap:
    """The two-sub-device manager of one mesh: dispatch on rows [0, 1), the shared expert on rows [1, grid_y)."""

    def __init__(self, mesh):
        self.mesh = mesh
        grid = mesh.compute_with_storage_grid_size()
        gx, gy = grid.x, grid.y
        rows = 1
        self.dispatch_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, rows - 1))})
        self.shared_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(gx - 1, gy - 1))})
        self.shared_grid = ttnn.CoreCoord(gx, gy - rows)
        self.manager = mesh.create_sub_device_manager(
            [ttnn.SubDevice([self.dispatch_cores]), ttnn.SubDevice([self.shared_cores])], 0
        )
        self.dispatch_id = ttnn.SubDeviceId(0)
        self.shared_id = ttnn.SubDeviceId(1)

    def load(self):
        if _capturing:
            _capturing[-1].split("load", self.manager)
        else:
            self.mesh.load_sub_device_manager(self.manager)

    def clear(self):
        if _capturing:
            _capturing[-1].split("clear", None)
        else:
            self.mesh.clear_loaded_sub_device_manager()


def overlap(mesh) -> Overlap:
    o = _overlaps.get(id(mesh))
    if o is None:
        o = _overlaps[id(mesh)] = Overlap(mesh)
    return o


class Trace:
    """A captured forward as trace segments with sub-device load / clear steps between them."""

    def __init__(self, mesh, cq_id=0):
        self.mesh, self.cq_id = mesh, cq_id
        self.program = []  # ("trace", tid) | ("load", manager) | ("clear", None)
        self.tid = None

    def split(self, kind, payload):
        ttnn.end_trace_capture(self.mesh, self.tid, cq_id=self.cq_id)
        self.program.append(("trace", self.tid))
        if kind == "load":
            self.mesh.load_sub_device_manager(payload)
        else:
            self.mesh.clear_loaded_sub_device_manager()
        self.program.append((kind, payload))
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=self.cq_id)

    @property
    def segments(self) -> int:
        return sum(1 for k, _ in self.program if k == "trace")


def begin_capture(mesh, cq_id=0):
    """ttnn.begin_trace_capture, or (MOE_OVERLAP on) a segmented Trace that the overlap's load / clear split."""
    if not enabled():
        return ttnn.begin_trace_capture(mesh, cq_id=cq_id)
    assert not _capturing, "one capture at a time"
    t = Trace(mesh, cq_id)
    t.tid = ttnn.begin_trace_capture(mesh, cq_id=cq_id)
    _capturing.append(t)
    return t


def end_capture(mesh, handle, cq_id=0) -> None:
    if not isinstance(handle, Trace):
        ttnn.end_trace_capture(mesh, handle, cq_id=cq_id)
        return
    ttnn.end_trace_capture(mesh, handle.tid, cq_id=handle.cq_id)
    handle.program.append(("trace", handle.tid))
    handle.tid = None
    _capturing.remove(handle)
    global last_segments
    last_segments = handle.segments


def execute(mesh, handle, cq_id=0, blocking=False) -> None:
    """Replays the capture: each segment non-blocking, the load / clear between them on the host (they only switch
    which manager the next segments run under; the device runs everything in enqueue order)."""
    if not isinstance(handle, Trace):
        ttnn.execute_trace(mesh, handle, cq_id=cq_id, blocking=blocking)
        return
    for kind, payload in handle.program:
        if kind == "trace":
            ttnn.execute_trace(mesh, payload, cq_id=handle.cq_id, blocking=False)
        elif kind == "load":
            mesh.load_sub_device_manager(payload)
        else:
            mesh.clear_loaded_sub_device_manager()
    if blocking:
        ttnn.synchronize_device(mesh)


def release(mesh, handle) -> None:
    """Releases every segment under the manager it was captured with."""
    if not isinstance(handle, Trace):
        ttnn.release_trace(mesh, handle)
        return
    loaded = False
    try:
        for kind, payload in handle.program:
            if kind == "trace":
                ttnn.release_trace(mesh, payload)
            elif kind == "load":
                mesh.load_sub_device_manager(payload)
                loaded = True
            else:
                mesh.clear_loaded_sub_device_manager()
                loaded = False
    finally:
        if loaded:
            mesh.clear_loaded_sub_device_manager()
        handle.program = []
