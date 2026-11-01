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
:class:`PerceptionFrame` buffers. The previous BEV is refilled by ``bev`` itself. With two command
queues, the images, by far the largest input, go through a staging buffer instead: queue 1 writes
the next frame's into it while queue 0 still runs this frame's ``features`` and ``bev``, and queue
0 copies it into the image buffer on device just before the next ``features``. Two events order
them: queue 0 waits for the write before the copy, and queue 1 for the copy before the next
write. Outputs are the traces' own buffers, overwritten by the next frame. Only those writes, the
staging copy, the counts' readback and the host compaction of the detections are outside the
traces; the copy writes into the image buffer, allocated before the first capture, and allocates
nothing.

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

# Command queue roles: every op, replay and readback on one, the images' upload on the other.
CQ_OPS = 0
CQ_INPUT = 1


class TTBEVFormerTrace:
    """Runs a :class:`TTBEVFormer` frame by frame from traces, taking over a live run's state.

    Attributes:
        captures: Seconds each capture took: the first one, then one per growth of the plan.
    """

    def __init__(self, model, device, host_image, frame, prev_bev, *, score_threshold=None, num_command_queues=1):
        """Capture the traces from a live run's state.

        Args:
            model: The :class:`TTBEVFormer`. It must have run two live frames, the second with a
                previous BEV, so that every program the traces record is compiled (the first frame
                skips the temporal path) and the planner is sized.
            device: The device the model runs on, opened with a trace region.
            host_image: A host tensor shaped like the model's image input, uploaded once to make
                the image buffer, and with two command queues the staging buffer.
            frame: The live run's :class:`PerceptionFrame`, whose buffers the traces read.
            prev_bev: The BEV the next frame reads as its previous one. The traces read and refill
                this buffer.
            score_threshold: Detections scoring below it are marked invalid, inside the trace.
            num_command_queues: 2 uploads the images on command queue 1, overlapped with the
                previous frame's device work; the device must have been opened with two.
        """
        assert num_command_queues in (1, 2), f"1 or 2 command queues, got {num_command_queues}"
        precondition = "run two live frames, the second with a previous BEV, before tracing"
        assert prev_bev is not None and model.planner.capacity > 0, precondition
        self.model = model
        self.device = device
        self.planner = model.planner
        self.score_threshold = score_threshold
        self.image = ttnn.to_device(host_image, device)
        # Allocated before any capture, so no trace's intermediates can sit on it.
        self.staging = ttnn.to_device(host_image, device) if num_command_queues == 2 else None
        # Whether an upload() awaits its frame, and with two queues its write's event.
        self._image_pending = False
        self._image_written = None
        self._image_read = None
        if self.staging is not None:
            try:
                ttnn.record_event(device, CQ_INPUT)
            except RuntimeError as error:
                raise ValueError("num_command_queues=2 needs a device opened with num_command_queues=2") from error
            # Compiled outside the captures, as every frame runs it live.
            ttnn.copy(self.staging, self.image)
            self._image_read = ttnn.record_event(device, CQ_OPS)
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
        try:
            self.capture_if_stale()
        except Exception:
            # A half-built runner is never released by its caller.
            self.release()
            raise

    @property
    def stale(self):
        """Whether the traces predate the planner's current generation and must be captured again."""
        return self._generation != self.planner.generation

    def upload(self, host_image):
        """Write a frame's images, a host tensor in the image buffer's layout: exactly once per
        frame, before its :meth:`__call__` and after the previous frame's. With two command queues
        the write goes to queue 1 and overlaps whatever the device still runs, so a pipelined
        caller uploads the next frame's images right after queuing this frame."""
        assert not self._image_pending, "run the uploaded frame before uploading the next"
        self._image_pending = True
        if self.staging is None:
            ttnn.copy_host_to_device_tensor(host_image, self.image, cq_id=CQ_OPS)
            return
        ttnn.wait_for_event(CQ_INPUT, self._image_read)
        ttnn.copy_host_to_device_tensor(host_image, self.staging, cq_id=CQ_INPUT)
        self._image_written = ttnn.record_event(self.device, CQ_INPUT)

    def _take_image(self):
        # The uploaded images into the image buffer, on the ops queue, and the input queue free to
        # write the next frame's.
        assert self._image_pending, "upload() each frame's images before running it"
        self._image_pending = False
        if self.staging is None:
            return
        ttnn.wait_for_event(CQ_OPS, self._image_written)
        ttnn.copy(self.staging, self.image)
        self._image_read = ttnn.record_event(self.device, CQ_OPS)

    def __call__(self, img_metas, host_frame=None):
        """Run one frame on the :meth:`upload`-ed images with their ``img_metas``. ``host_frame`` is
        the frame's :meth:`TTBEVFormer.host_frame` if the caller prepared it ahead (for example on a
        host thread), or None to prepare it here. Returns the
        device ``(scores, labels, boxes, valid)`` of :meth:`TTBEVFormer.detect`, valid until the
        next call; synchronize before reading them. A frame that finds the traces stale, or grows
        the plan, runs live; :meth:`capture_if_stale` then captures them again."""
        self.model.refill_frame(img_metas, self.frame, host_frame)
        if self.stale:
            return self._live()
        ttnn.execute_trace(self.device, self.trace_ids["cameras"], cq_id=CQ_OPS, blocking=False)
        visibility = self.planner.read_counts(self.frame.cameras.visibility)
        self._take_image()
        ttnn.execute_trace(self.device, self.trace_ids["features"], cq_id=CQ_OPS, blocking=False)
        self.planner.size(visibility)
        if not self.stale:
            ttnn.execute_trace(self.device, self.trace_ids["bev"], cq_id=CQ_OPS, blocking=False)
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
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=CQ_OPS)
            try:
                forward()
            finally:
                # Also on failure: a device left capturing rejects every later command.
                ttnn.end_trace_capture(self.device, trace_id, cq_id=CQ_OPS)
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
        self._take_image()
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
