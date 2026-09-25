# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace islands: the position-independent slices of a V4 prefill block (mHC sites, norms, MoE) captured ONCE as a
segmented ttnn trace and replayed every chunk, while the chunk-position-dependent attention core stays eager.

Why: the eager block is host-dispatch bound (DS4F-0246: ~0.12 ms per program, 571-780 programs per layer; the device
idles ~55-60%). The mHC sites alone are ~280 of those programs and know nothing about the chunk position, so they trace
once per layer and serve every chunk of every request; the attention (slice starts, fill_cache offsets, tail matrices
keyed by the entry count) is the part whose program arguments change per chunk (DS4F-0247).

Rules a caller must keep (the trace bakes ADDRESSES):
  * an island's inputs are persistent device buffers (owned by this island, or the persistent outputs of another
    island captured earlier); the block copies each chunk's live activations into them with ``ttnn.copy`` before replay;
  * an island's outputs are persistent for its lifetime -- consumers never deallocate them;
  * no eager tensor may stay alive across a replay unless it was allocated before every island was captured: the
    trace's intermediates land in whatever DRAM was free at capture time, and a later eager allocation there is
    overwritten by the replay. The block frees its eager temporaries (the attention output) before the next replay;
  * every island's programs must have been compiled once (an eager warm-up) before capture.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import ttnn
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController


class TraceIsland:
    def __init__(self, mesh_device, fn: Callable, inputs: Sequence, *, moe=None, name: str = ""):
        self.mesh_device = mesh_device
        self.fn = fn
        self.inputs = list(inputs)
        self.moe = moe  # a TtMoe inside fn: its sub-device swaps split the capture (SubDeviceTraceController)
        self.name = name
        self.outputs: Optional[tuple] = None
        self.controller: Optional[SubDeviceTraceController] = None

    def capture(self) -> tuple:
        """Run ``fn(*inputs)`` once under capture; its return value becomes the persistent outputs."""
        assert self.controller is None, f"island {self.name} already captured"
        controller = SubDeviceTraceController(self.mesh_device)
        if self.moe is not None:
            self.moe.set_trace_controller(controller)
        try:
            controller.begin_capture()
            out = self.fn(*self.inputs)
            controller.end_capture()
        finally:
            if self.moe is not None:
                self.moe.set_trace_controller(None)
        self.controller = controller
        self.outputs = tuple(out) if isinstance(out, (list, tuple)) else (out,)
        return self.outputs

    def replay(self) -> tuple:
        """Enqueue every captured segment NON-blocking (the sub-device load/clear between segments are host-side
        registry switches); the caller's next eager op queues behind them on the same CQ."""
        c = self.controller
        assert c is not None, f"island {self.name} not captured"
        for kind, payload in c._program:
            if kind == c._TRACE:
                ttnn.execute_trace(self.mesh_device, payload, cq_id=c.cq_id, blocking=False)
            elif kind == c._LOAD:
                self.mesh_device.load_sub_device_manager(payload)
            elif kind == c._CLEAR:
                self.mesh_device.clear_loaded_sub_device_manager()
            else:
                raise AssertionError(f"island {self.name}: unexpected segment kind {kind}")
        return self.outputs

    @property
    def num_segments(self) -> int:
        return 0 if self.controller is None else self.controller.num_segments

    def release(self) -> None:
        if self.controller is not None:
            self.controller.release()
            self.controller = None


def copy_into(dst, src) -> None:
    """``dst[:] = src`` on device (same shape / dtype / layout)."""
    ttnn.copy(src, dst)
