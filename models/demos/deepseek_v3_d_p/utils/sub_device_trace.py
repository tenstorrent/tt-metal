# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Segmented ttnn-trace capture across sub-device-manager swaps.

Problem: ttnn forbids loading/clearing a sub-device manager *inside* a trace capture
(`begin_trace_capture`..`end_trace_capture`) — it resets worker state. But the MoE wants to keep the
shared-expert/dispatch overlap, which loads a 2-sub-device manager around that region and clears it
after. So a single trace cannot span the whole forward.

Solution: capture the forward as *several* traces, split exactly at the load/clear points, and perform
the load/clear on the host *between* the trace segments (legal — we are not capturing at that instant).
At replay we walk the recorded program, executing each trace segment and doing the load/clear between
them. Validated bit-exact in scratch PoC (3 chained traces, load/clear between captures and replays).

Usage (capture once, replay many) — the forward runs ONCE during capture; this controller chops it:

    controller = SubDeviceTraceController(mesh_device)
    transformer.set_trace_controller(controller)   # MoE consults it for load/clear
    transformer.forward(...)                         # WARMUP (controller idle) -> compiles programs
    controller.begin_capture()
    transformer.forward(...)                         # captured + auto-split at each load/clear
    controller.end_capture()
    ...                                              # controller.trace_bytes() for memory
    for _ in range(n): controller.replay()           # each replay = full segmented forward
    controller.release()
    transformer.set_trace_controller(None)

Boundary tensors (produced in one segment, read by the next) stay valid because the forward keeps them
referenced across the end/begin split; trace replay reproduces them at the same addresses.
"""

import ttnn


class SubDeviceTraceController:
    """Drives multi-segment trace capture/replay around sub-device-manager load/clear calls.

    MoE.forward calls `sub_device_load(mgr_id)` / `sub_device_clear()` instead of touching the mesh
    device directly. This object decides what those mean based on its mode:
      - idle (not capturing): pass straight through to the real device load/clear (eager behavior).
      - capturing: end the in-progress trace, do the real host load/clear, begin the next trace —
        i.e. split the capture here and record the boundary action.
    """

    # Program step kinds.
    _TRACE = "trace"
    _LOAD = "load"
    _CLEAR = "clear"
    _ACK = "ack"  # per-layer migration ack (host shm bump): split the capture here, call the callback at replay

    def __init__(self, mesh_device, cq_id=0):
        self.mesh_device = mesh_device
        self.cq_id = cq_id
        self._program = []  # ordered (kind, payload): (_TRACE, tid)|(_LOAD, mgr_id)|(_CLEAR, None)|(_ACK, layer_idx)
        self._current_tid = None
        self._capturing = False
        # Optional per-layer ack callback (runner: layer_ack_channel.inject(1)). When set, MLA routes its
        # on_layer_complete through layer_ack() so the migration ack stays correct under trace replay: the
        # capture splits the trace at the ack point (a host shm bump cannot be inside a trace), and replay
        # calls the callback BETWEEN the two trace segments (after the first segment's KV writes flush,
        # before the next). None => no ack boundaries (test path: pure sub-device-swap segmentation).
        self._on_layer_complete = None

    def set_layer_ack_callback(self, on_layer_complete):
        """Register the per-layer migration-ack callback. See layer_ack()."""
        self._on_layer_complete = on_layer_complete

    def has_layer_ack(self):
        return self._on_layer_complete is not None

    # ------------------------------------------------------------------ capture
    def begin_capture(self):
        """Open the first trace segment. The next forward() will be recorded and auto-split."""
        assert not self._capturing, "already capturing"
        self._program = []
        self._capturing = True
        self._current_tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=self.cq_id)

    def end_capture(self):
        """Close the final trace segment."""
        assert self._capturing, "begin_capture() was not called"
        ttnn.end_trace_capture(self.mesh_device, self._current_tid, cq_id=self.cq_id)
        self._program.append((self._TRACE, self._current_tid))
        self._current_tid = None
        self._capturing = False

    # ----------------------------------------------------- MoE-facing hooks
    def sub_device_load(self, sd_manager_id):
        """MoE entered the overlap region. Capturing -> split + record + real load; else just load."""
        if self._capturing:
            self._split(self._LOAD, sd_manager_id)
        else:
            self.mesh_device.load_sub_device_manager(sd_manager_id)

    def sub_device_clear(self):
        """MoE left the overlap region. Capturing -> split + record + real clear; else just clear."""
        if self._capturing:
            self._split(self._CLEAR, None)
        else:
            self.mesh_device.clear_loaded_sub_device_manager()

    def layer_ack(self, layer_idx):
        """MLA finished this layer's KV write (post zero-pad). Capturing -> split + record the ack
        boundary (NO host action during capture: the capture pass is not a real chunk, so we must not
        bump the ack counter). Eager (not capturing) -> call the callback directly (matches the
        non-trace runner). At replay the ack callback fires between the two trace segments. No-op if no
        callback is registered."""
        if self._on_layer_complete is None:
            return
        if self._capturing:
            self._split(self._ACK, layer_idx)
        else:
            # Eager (idle controller): flush this layer's KV before the ack, same as the non-trace path —
            # the migration worker reads it out-of-band from the CQ.
            ttnn.synchronize_device(self.mesh_device)
            self._on_layer_complete(layer_idx)

    def _split(self, kind, payload):
        # Close the current segment, perform the real host action (load/clear only; ACK has no
        # capture-time action), open the next segment.
        ttnn.end_trace_capture(self.mesh_device, self._current_tid, cq_id=self.cq_id)
        self._program.append((self._TRACE, self._current_tid))
        if kind == self._LOAD:
            self.mesh_device.load_sub_device_manager(payload)
        elif kind == self._CLEAR:
            self.mesh_device.clear_loaded_sub_device_manager()
        # kind == _ACK: no host action at capture time (the ack fires at replay, between segments).
        self._program.append((kind, payload))
        self._current_tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=self.cq_id)

    # ------------------------------------------------------------------ replay
    def replay(self):
        """Run the whole segmented forward: execute each trace, load/clear between segments.

        Segments are enqueued NON-blocking on one CQ: the device runs them in enqueue order, and the
        load/clear between them are host-side registry switches (they pick which manager's trace the next
        execute_trace looks up; per-manager trace buffers persist), so they do not need the prior segment
        to have finished on device. We force a device sync in only two places:
          - BEFORE each per-layer migration ack — the migration worker reads the just-written KV over NoC,
            out-of-band from the CQ, so that segment's writes must be on-device first; and
          - ONCE after the last segment — so the chunk's KV is fully written before replay() returns.
        This replaces the old block-every-segment (which serialized dispatch for no reason but the ack)."""
        assert not self._capturing, "still capturing"
        assert self._program, "nothing captured"
        for kind, payload in self._program:
            if kind == self._TRACE:
                ttnn.execute_trace(self.mesh_device, payload, cq_id=self.cq_id, blocking=False)
            elif kind == self._LOAD:
                self.mesh_device.load_sub_device_manager(payload)
            elif kind == self._CLEAR:
                self.mesh_device.clear_loaded_sub_device_manager()
            else:  # _ACK: per-layer migration ack (callback set => non-None here). Flush the preceding
                # segment(s) so the out-of-band migration worker sees post-write KV, THEN fire the ack.
                ttnn.synchronize_device(self.mesh_device)
                self._on_layer_complete(payload)
        # Ack path: flush the tail (everything after the last ack) so the migration worker's out-of-band
        # NoC read sees the last layer's KV. NO-ACK (pipeline / standalone): do NOT sync per chunk — the
        # segments are enqueued in order on one CQ, so chunk N's KV write completes before chunk N+1's
        # replay reads it (and before the D2D send reads _trace_output), and the driver's end-of-loop drain
        # (_drain_and_log_e2e) does the final flush before any readback. Syncing here per chunk serialized
        # the traced pipeline and negated the cross-chunk overlap the untraced async path gets.
        if self._on_layer_complete is not None:
            ttnn.synchronize_device(self.mesh_device)

    # ------------------------------------------------------------------ stats / cleanup
    @property
    def num_segments(self):
        return sum(1 for kind, _ in self._program if kind == self._TRACE)

    def trace_bytes(self):
        """Total device memory used by all captured trace segments (bytes per device)."""
        mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
        return mv.total_bytes_allocated_per_bank * mv.num_banks

    def release(self):
        """Release every captured trace. Safe to call repeatedly."""
        for kind, payload in self._program:
            if kind == self._TRACE:
                ttnn.release_trace(self.mesh_device, payload)
        self._program = []
