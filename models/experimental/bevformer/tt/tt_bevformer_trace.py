# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BEVFormer frames replayed from traces instead of dispatched op by op.

A frame is three traces around the one value the host must see, the cameras' visible counts that
size the spatial cross-attention's plan:

* ``cameras``: the frame's camera projection and visibility.
* ``features``: the backbone and the FPN. Queued right after the counts' readback, so the counts
  come back while it runs.
* ``bev``: the plan at the planner's capacity, the perception transformer, the head and the
  detections, ending with a copy of the frame's BEV into the buffer the next frame reads as its
  previous BEV.

Between ``features`` and ``bev`` the planner sizes the plan from the counts. While its
``generation`` holds, ``bev`` would record the same ops on the same constants and shapes, so it
replays; a frame no camera sees replays too, its plan all padding. When the planner grows, the
frame runs ``bev`` live at the new shapes and the traces are stale; so is a frame that starts with
the planner changed since the capture, which then runs live from the start. A capture records only
programs a live run has compiled, so :meth:`capture_if_stale` captures all three again only after
such a frame, and a caller can keep it out of the frame's timing. All three, because the planner's
new constants were allocated after the old captures, where an old trace's intermediates may sit.

Every input is a buffer the host refills in place on command queue 0, in order with the replays
that read it: the images (:meth:`TTBEVFormerTrace.upload`) and the frame's
:class:`PerceptionFrame` buffers. The previous BEV is refilled by ``bev`` itself. Outputs are the
traces' own buffers, overwritten by the next frame. Only those writes, the counts' readback and the
host compaction of the detections are outside the traces.

The traces share memory safely only because they always replay in capture order. Each is captured
while the earlier ones' outputs are alive, so its buffers never overlap them, but they may overlap
the earlier traces' intermediates. A later trace's outputs may therefore be overwritten by the next
frame's earlier traces: ``features`` levels and ``bev`` detections are consumed within the frame,
which is why the detections are valid only until the next call. Nothing else may allocate device
memory between replays, since it could land on a trace's intermediates, except the planner's
constants on growth: ``cameras`` and ``features`` outputs are alive then, and the old traces are
not replayed again.
"""

import time

import ttnn


class TTBEVFormerTrace:
    """Runs a :class:`TTBEVFormer` frame by frame from traces, taking over a live run's state.

    Attributes:
        captures: Seconds each capture took: the first one, then one per growth of the plan.
    """

    def __init__(self, model, device, host_image, frame, prev_bev, *, score_threshold=None):
        """Capture the traces from a live run's state.

        Args:
            model: The :class:`TTBEVFormer`. It must have run two live frames, the second with a
                previous BEV, so that every program the traces record is compiled (the first frame
                skips the temporal path) and the planner is sized.
            device: The device the model runs on, opened with a trace region.
            host_image: A host tensor shaped like the model's image input, uploaded once to make
                the image buffer.
            frame: The live run's :class:`PerceptionFrame`, whose buffers the traces read.
            prev_bev: The BEV the next frame reads as its previous one. The traces read and refill
                this buffer.
            score_threshold: Detections scoring below it are marked invalid, inside the trace.
        """
        precondition = "run two live frames, the second with a previous BEV, before tracing"
        assert prev_bev is not None and model.planner.capacity > 0, precondition
        self.model = model
        self.device = device
        self.planner = model.planner
        self.score_threshold = score_threshold
        self.image = ttnn.to_device(host_image, device)
        self.frame = frame
        self.prev_bev = prev_bev
        self.trace_ids = {}
        self.levels = self.outputs = None
        self.captures = []
        self._generation = None
        # The planner generation the live run compiled every traced program at.
        self._compiled = self.planner.generation
        # The live run never copies its BEV out, so that one program is compiled here, into a
        # scratch tensor: the previous BEV must stay the live run's.
        scratch = ttnn.clone(prev_bev)
        ttnn.copy(prev_bev, scratch)
        ttnn.deallocate(scratch)
        self.capture_if_stale()

    @property
    def stale(self):
        """Whether the traces predate the planner's current generation and must be captured again."""
        return self._generation != self.planner.generation

    def upload(self, host_image):
        """Write the next frame's images, a host tensor in the image buffer's layout."""
        ttnn.copy_host_to_device_tensor(host_image, self.image)

    def __call__(self, img_metas):
        """Run one frame on the :meth:`upload`-ed images with their ``img_metas``. Returns the
        device ``(scores, labels, boxes, valid)`` of :meth:`TTBEVFormer.detect`, valid until the
        next call; synchronize before reading them. A frame that finds the traces stale, or grows
        the plan, runs live; :meth:`capture_if_stale` then captures them again."""
        self.model.refill_frame(img_metas, self.frame)
        if self.stale:
            return self._live()
        ttnn.execute_trace(self.device, self.trace_ids["cameras"], cq_id=0, blocking=False)
        visibility = self.planner.read_counts(self.frame.cameras.visibility)
        ttnn.execute_trace(self.device, self.trace_ids["features"], cq_id=0, blocking=False)
        self.planner.size(visibility)
        if not self.stale:
            ttnn.execute_trace(self.device, self.trace_ids["bev"], cq_id=0, blocking=False)
            return self.outputs
        return self._live_bev()

    def capture_if_stale(self):
        """Capture the three traces again if the plan has changed since the last capture and a live
        frame has run at its current shapes."""
        if not self.stale or self._compiled != self.planner.generation:
            return
        ttnn.synchronize_device(self.device)
        start = time.perf_counter()
        self.release()

        def capture_cameras():
            self.model.project_cameras(self.frame)

        def capture_features():
            self.levels = self.model.features(self.image)

        def capture_bev():
            self.outputs = self._bev()

        for name, forward in (("cameras", capture_cameras), ("features", capture_features), ("bev", capture_bev)):
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            try:
                forward()
            finally:
                # Also on failure: a device left capturing rejects every later command.
                ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
                self.trace_ids[name] = trace_id
        self._generation = self.planner.generation
        self.captures.append(time.perf_counter() - start)

    def release(self):
        """Release the traces; :meth:`capture_if_stale` captures them again."""
        for trace_id in self.trace_ids.values():
            ttnn.release_trace(self.device, trace_id)
        self.trace_ids = {}
        self._generation = None

    def _live(self):
        visibility = self.planner.read_counts(self.model.project_cameras(self.frame).visibility)
        self.levels = self.model.features(self.image)
        self.planner.size(visibility)
        return self._live_bev()

    def _live_bev(self):
        outputs = self._bev()
        self._compiled = self.planner.generation
        return outputs

    def _bev(self):
        cameras = self.frame.cameras
        self.model.build_plan(self.frame)
        cls_scores, bbox_preds, bev_embed = self.model.bev(self.levels, self.frame, prev_bev=self.prev_bev)
        # The plan is this frame's; the next frame, or capture, builds its own on the projection.
        self.frame.cameras = cameras
        scores, labels, _, boxes, valid = self.model.detect(cls_scores, bbox_preds, self.score_threshold)
        # Last: everything above has read the previous BEV.
        ttnn.copy(bev_embed, self.prev_bev)
        return scores, labels, boxes, valid
