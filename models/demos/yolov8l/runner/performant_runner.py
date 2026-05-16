# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import torch

import ttnn
from models.demos.yolov8l.runner.performant_runner_infra import YOLOv8lPerformanceRunnerInfra

try:
    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class YOLOv8lPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size,
        inp_h=None,
        inp_w=None,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        mesh_mapper=None,
        mesh_composer=None,
        weights_mesh_mapper=None,
        model_location_generator=None,
        compact_output=False,
        staging_ring=1,
        uint8_input=False,
    ):
        self.device = device
        self.mesh_mapper = mesh_mapper
        self.mesh_composer = mesh_composer
        self.weights_mesh_mapper = weights_mesh_mapper
        self.compact_output = compact_output
        # When True, the host produces a uint8 TILE-layout input; PCIe carries
        # half the bytes of bf16 RM.  The trace prepends typecast(u8→bf16) +
        # multiply(1/255) + untilize before the existing reshard so the model
        # graph (bf16 RM) is unchanged.
        self.uint8_input = uint8_input
        # K dram_staging+host_staging pairs.  K=1 preserves the original behavior
        # where copy(N+1) on CQ0 stalls waiting for d2h(N-1) on CQ1.  K>=2 lets
        # CQ0 advance independently of CQ1 — capped by trace_time vs d2h_time.
        self.staging_ring = max(1, int(staging_ring))
        self.runner_infra = YOLOv8lPerformanceRunnerInfra(
            device,
            device_batch_size,
            inp_h=inp_h,
            inp_w=inp_w,
            mesh_mapper=self.mesh_mapper,
            mesh_composer=self.mesh_composer,
            weights_mesh_mapper=self.weights_mesh_mapper,
            model_location_generator=model_location_generator,
        )
        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra._setup_dram_sharded_input(device, uint8_input=uint8_input)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self._capture_yolov8l_trace_2cqs()

    def _convert_tensor_to_input_config(self, tensor):
        """Convert tensor to the appropriate memory configuration for input.

        For ``uint8_input=True``: tensor is uint8 TILE in interleaved DRAM
        (h2d target).  Apply ``typecast(u8→bf16) + multiply(1/255) + untilize``
        producing bf16 RM in DRAM with the conv's expected shard spec.

        For bf16 path: identity (conv reads from DRAM-sharded RM directly).
        """
        if not self.uint8_input:
            return tensor
        bf16_tile = ttnn.typecast(tensor, ttnn.bfloat16)
        bf16_norm = ttnn.multiply(bf16_tile, 1.0 / 255.0)
        ttnn.deallocate(bf16_tile)
        bf16_rm = ttnn.untilize(
            bf16_norm,
            use_multicore=True,
            memory_config=self.runner_infra._bf16_dram_sharded_config,
        )
        ttnn.deallocate(bf16_norm)
        return bf16_rm

    def _reduce_classes(self, rm_output):
        """Compact YOLO output [B, 84, A] → [B, 6, A] via on-device class reduction.

        Channels 0..3 = box (cx, cy, w, h) preserved; channels 4..83 (80 classes)
        collapsed to channel 4 = max conf and channel 5 = argmax class id.

        Implementation note: a separate ttnn.argmax + uint32→bf16 typecast chain
        runs ~15ms at this shape on BH; ttnn.topk(k=1) returns value+index in a
        single ~0.5ms op so we use that.

        Cuts D2H volume 14× — the dominant per-frame cost in the demo pipeline.
        """
        anchors = int(rm_output.shape[-1])
        box = ttnn.slice(rm_output, [0, 0, 0], [1, 4, anchors])
        cls = ttnn.slice(rm_output, [0, 4, 0], [1, 84, anchors])
        cls_t = ttnn.to_layout(cls, ttnn.TILE_LAYOUT)
        cls_tT = ttnn.transpose(cls_t, 1, 2)  # [1, A, 80] tile (argmax/topk need last dim)
        vals, idx = ttnn.topk(cls_tT, k=1, dim=-1)  # [1, A, 1] bf16 / uint16 tile
        vals_T = ttnn.transpose(vals, 1, 2)  # [1, 1, A] bf16 tile
        vals_rm = ttnn.to_layout(vals_T, ttnn.ROW_MAJOR_LAYOUT)
        idx_bf = ttnn.typecast(idx, ttnn.bfloat16)  # uint16→bf16 works directly (unlike uint32)
        idx_T = ttnn.transpose(idx_bf, 1, 2)  # [1, 1, A] bf16 tile
        idx_rm = ttnn.to_layout(idx_T, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.concat([box, vals_rm, idx_rm], dim=1)

    def _capture_yolov8l_trace_2cqs(self):
        # Initialize the op event so we can write
        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT; use DRAM input like trace capture (L1-staged 1280² input clashes with conv CBs).
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self._convert_tensor_to_input_config(self.tt_image_res)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self._convert_tensor_to_input_config(self.tt_image_res)
        dram_spec = self.runner_infra.input_tensor.spec  # For trace
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        if not self.compact_output:
            self.runner_infra.validate()

        # Untilize detection output on-device (TILE → ROW_MAJOR) and strip
        # tile-alignment padding in one fused op.  untilize_with_unpadding
        # produces a truly unpadded buffer (e.g. [1,84,33600] instead of
        # [1,96,33600]), reducing D2H volume by ~12.5%.
        tile_output = self.runner_infra.output_tensor[0]
        _logical = list(tile_output.shape)
        _padded = list(tile_output.padded_shape)
        self._needs_unpad = _padded != _logical
        if self._needs_unpad:
            self._output_end = tuple(s - 1 for s in _logical)  # inclusive end
            rm_output = ttnn.untilize_with_unpadding(
                tile_output,
                output_tensor_end=self._output_end,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            print(
                f"[runner] untilize_with_unpadding: {_padded} → {_logical} "
                f"(saves {(_padded[-2]-_logical[-2])*_padded[-1]*2/1024:.0f} KB/shard)",
                flush=True,
            )
        else:
            rm_output = ttnn.untilize(tile_output, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tile_output)
        if self.compact_output:
            compact_rm = self._reduce_classes(rm_output)
            ttnn.deallocate(rm_output)
            self.runner_infra.output_tensor[0] = compact_rm
            print(
                f"[runner] compact_output: 84-channel → 6-channel "
                f"(box[4] + max_conf[1] + argmax_id[1]) — 14× D2H reduction",
                flush=True,
            )
        else:
            self.runner_infra.output_tensor[0] = rm_output

        # Pre-allocate K staging buffer pairs for pipelined D2H/compute overlap.
        # Must happen BEFORE trace capture so the trace allocator avoids
        # these addresses — post-trace allocations get corrupted when trace
        # execution overwrites intermediate buffers at reused addresses.
        K = self.staging_ring
        self.dram_staging_ring = []
        self.host_staging_ring = []
        self.staging_ready_event_ring = []
        self.read_done_event_ring = []
        for _ in range(K):
            ds = ttnn.clone(
                self.runner_infra.output_tensor[0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            hs = ttnn.allocate_tensor_on_host(ds.spec, self.device)
            self.dram_staging_ring.append(ds)
            self.host_staging_ring.append(hs)
            # Initial events are "already done" so the first writer doesn't
            # block waiting for a non-existent prior d2h.
            self.staging_ready_event_ring.append(ttnn.record_event(self.device, 0))
            self.read_done_event_ring.append(ttnn.record_event(self.device, 1))
        # Backwards-compatible single-buffer aliases (slot 0).  External
        # callers that read these properties keep working.
        self.dram_staging = self.dram_staging_ring[0]
        self.host_staging = self.host_staging_ring[0]
        self.staging_ready_event = self.staging_ready_event_ring[0]
        self.read_done_event = self.read_done_event_ring[0]
        self._stg_write_idx = 0  # next slot to write (incremented after copy in submit)
        self._last_read_idx = -1  # most recent slot read by pcie_d2h
        self._pipeline_frame = 0
        self._compose_timing = 0.0
        # Periodic device-compute measurement (Option 4 — event-based).
        # Set _compute_measure_every <= 0 to disable.  At 100-frame cadence
        # avg overhead is ~0.3 ms/frame (one ~30ms host-block per 100 frames).
        self._compute_measure_every = 100
        self._last_compute_ms: float | None = None
        if K > 1:
            print(f"[runner] staging_ring=K={K} (CQ0/CQ1 decoupled)", flush=True)

        # Pre-allocate compose output buffer sized for PHYSICAL shape (includes
        # tile-alignment padding).  batch_to_torch(physical=True) copies the full
        # physical buffer contiguously (24 large memcpy calls instead of 2016
        # strided ones).  Callers slice [:, :log_h, :log_w] to get logical data.
        # NOTE: physical=False was tested and is 2x slower (7.8ms vs 4.4ms)
        # due to the 2016 strided copy overhead.
        n_devices = len(ttnn.get_device_tensors(self.host_staging))
        _padded = self.dram_staging.padded_shape
        _logical = self.dram_staging.shape
        _phys_h = int(_padded[-2]) if len(_padded) >= 2 else 1
        _phys_w = int(_padded[-1])
        _log_h = int(_logical[-2]) if len(_logical) >= 2 else 1
        _log_w = int(_logical[-1])
        self._phys_per_shard = (_phys_h, _phys_w)
        self._log_per_shard = (_log_h, _log_w)
        self._compose_buf = torch.empty(n_devices, _phys_h, _phys_w, dtype=torch.bfloat16)
        self._compose_physical = _phys_h != _log_h or _phys_w != _log_w
        print(
            f"[runner] staging logical={list(_logical)} padded={list(_padded)} "
            f"physical=({_phys_h}×{_phys_w}) logical=({_log_h}×{_log_w}) "
            f"compose_physical={self._compose_physical} "
            f"compose_buf={list(self._compose_buf.shape)}",
            flush=True,
        )

        # Capture trace
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self._convert_tensor_to_input_config(self.tt_image_res)
        self.op_event = ttnn.record_event(self.device, 0)
        # Deallocate output to ensure input gets same address after trace
        self.runner_infra.dealloc_output()
        # uint8 path: deallocate the pre-trace converted tensor so the trace
        # allocator reuses its address for the in-trace conversion output.
        if self.uint8_input:
            ttnn.deallocate(self.runner_infra.input_tensor)
        trace_input_addr = (
            self.tt_image_res.buffer_address() if self.uint8_input else self.runner_infra.input_tensor.buffer_address()
        )
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        # uint8 path: re-run conversion INSIDE the trace so each replay reads
        # fresh uint8 data from tt_image_res (h2d updates each frame).
        if self.uint8_input:
            self.runner_infra.input_tensor = self._convert_tensor_to_input_config(self.tt_image_res)
        self.runner_infra.run()
        # On-device output processing inside trace — replayed every frame
        tile_output_traced = self.runner_infra.output_tensor[0]
        if self._needs_unpad:
            rm_traced = ttnn.untilize_with_unpadding(
                tile_output_traced,
                output_tensor_end=self._output_end,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            rm_traced = ttnn.untilize(tile_output_traced, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tile_output_traced)
        if self.compact_output:
            compact_traced = self._reduce_classes(rm_traced)
            ttnn.deallocate(rm_traced)
            self.runner_infra.output_tensor[0] = compact_traced
        else:
            self.runner_infra.output_tensor[0] = rm_traced
        self.input_tensor = ttnn.allocate_tensor_on_device(dram_spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        # assert trace_input_addr == self.input_tensor.buffer_address()

    def _setup_staging_buffer(self):
        """No-op: staging buffer is pre-allocated during trace capture.

        Kept for backward compatibility with callers that invoke this method.
        """

    def _execute_yolov8l_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        # uint8 path: the trace itself converts u8→bf16+/255+untilize+reshard
        # so we skip the per-frame reshard (and tt_image_res is uint8 TILE,
        # incompatible with input_mem_config which is bf16 RM L1 sharded).
        if not self.uint8_input:
            # TODO: Add in place support to ttnn to_memory_config
            self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)

        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        return self.runner_infra.output_tensor

    # ------------------------------------------------------------------
    # Original synchronous API (unchanged)
    # ------------------------------------------------------------------

    def run(self, torch_input_tensor=None):
        t0 = time.perf_counter()
        if torch_input_tensor is None:
            tt_inputs_host = None
        else:
            tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(
                self.device, torch_input_tensor, uint8_input=self.uint8_input
            )
        t1 = time.perf_counter()
        result = self._execute_yolov8l_trace_2cqs_inference(tt_inputs_host)
        t2 = time.perf_counter()
        self.last_timing = {
            "host_prep_ms": (t1 - t0) * 1000,
            "h2d_and_trace_ms": (t2 - t1) * 1000,
        }
        return result

    # ------------------------------------------------------------------
    # Pipelined API: overlaps D2H(N-1) with compute(N)
    # ------------------------------------------------------------------

    def prepare_input(self, torch_input_tensor):
        """CPU-only host prep: convert torch tensor to ttnn host buffer.

        Uses ``from_host_shards`` with per-shard ``from_torch`` calls.
        The 24 ``from_torch`` calls are the bottleneck (~18ms in a thread),
        but this format produces hugepage-backed shards that are required
        for efficient D2H (22ms vs 35ms with mesh_mapper format).

        Safe to call while D2H is in progress on CQ1 — no device commands.
        Returns the host tensor for use in ``enqueue_frame``.
        """
        t0 = time.perf_counter()
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(
            self.device, torch_input_tensor, uint8_input=self.uint8_input
        )
        self._last_host_prep_ms = (time.perf_counter() - t0) * 1000
        return tt_inputs_host

    def enqueue_frame(self, tt_inputs_host=None):
        """Queue H2D + staging copy + reshard + trace.  Non-blocking.

        If ``tt_inputs_host`` is None, uses the last prepared input.
        Must be called after ``prepare_input`` (or pass the result directly).

        DEBUG: set runner._skip_h2d = True to bypass h2d (for perf-ceiling
        measurement only — model runs on stale device-side input).
        """
        t0 = time.perf_counter()
        # Queue H2D on CQ1
        ttnn.wait_for_event(1, self.op_event)
        t1 = time.perf_counter()
        if not getattr(self, "_skip_h2d", False):
            ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        t2 = time.perf_counter()
        self.write_event = ttnn.record_event(self.device, 1)

        # CQ0: wait for H2D, then copy prev output → staging, then compute
        ttnn.wait_for_event(0, self.write_event)
        t3 = time.perf_counter()

        # Copy previous output to next staging-ring slot.  Per-slot
        # read_done_event lets CQ0 advance K-1 frames before having to wait
        # for CQ1 to release the slot.
        if self._pipeline_frame > 0:
            slot = self._stg_write_idx % self.staging_ring
            ttnn.wait_for_event(0, self.read_done_event_ring[slot])
            ttnn.copy(self.runner_infra.output_tensor[0], self.dram_staging_ring[slot])
            self.staging_ready_event_ring[slot] = ttnn.record_event(self.device, 0)
            self.staging_ready_event = self.staging_ready_event_ring[slot]  # alias
            self._stg_write_idx += 1
        t4 = time.perf_counter()

        # Reshard input + start trace (all on CQ0, after reshard-to-staging).
        # uint8 path: skip — conversion + reshard live inside the trace.
        if not self.uint8_input:
            self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        t5 = time.perf_counter()
        self.op_event = ttnn.record_event(self.device, 0)

        # Periodic device-compute timing (Option 4): record events around the
        # trace, host-block on both, diff = on-chip trace runtime.  Adds
        # ~queue_drain + compute_time of host wait every Nth frame; net
        # impact <0.2ms/frame at MEASURE_EVERY=100.
        measure_compute = self._pipeline_frame % self._compute_measure_every == 0
        compute_start_evt = ttnn.record_event(self.device, 0) if measure_compute else None

        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        if measure_compute:
            compute_end_evt = ttnn.record_event(self.device, 0)
            ttnn.event_synchronize(compute_start_evt)
            t_compute_start = time.perf_counter()
            ttnn.event_synchronize(compute_end_evt)
            self._last_compute_ms = (time.perf_counter() - t_compute_start) * 1000

        t_queue = (time.perf_counter() - t0) * 1000
        self._pipeline_frame += 1
        self.last_timing = {
            "host_prep_ms": self._last_host_prep_ms,
            "queue_ms": t_queue,
            "wait_op_ms": (t1 - t0) * 1000,
            "h2d_ms": (t2 - t1) * 1000,
            "wait_h2d_ms": (t3 - t2) * 1000,
            "staging_ms": (t4 - t3) * 1000,
            "reshard_ms": (t5 - t4) * 1000,
        }
        if self._last_compute_ms is not None:
            self.last_timing["compute_ms"] = self._last_compute_ms

    def submit(self, torch_input_tensor):
        """Host-prep + queue H2D + reshard-to-staging + compute.  Non-blocking.

        Convenience wrapper around ``prepare_input`` + ``enqueue_frame``.
        """
        tt_inputs_host = self.prepare_input(torch_input_tensor)
        self.enqueue_frame(tt_inputs_host)

    def get_result(self, mesh_composer=None):
        """D2H previous frame's staging buffer on CQ1.  Blocks host.

        While the host blocks here, CQ0 is executing the current frame's
        trace in parallel — this is where the overlap happens.

        Returns ``None`` on the very first call (no previous frame yet).
        """
        if self._pipeline_frame <= 1:
            return None

        t0 = time.perf_counter()
        composer = mesh_composer or self.mesh_composer
        ttnn.wait_for_event(1, self.staging_ready_event)
        t_wait = time.perf_counter()

        # D2H into pre-allocated host buffer
        ttnn.copy_device_to_host_tensor(self.dram_staging, self.host_staging, blocking=True, cq_id=1)
        self.read_done_event = ttnn.record_event(self.device, 1)
        t_pcie = time.perf_counter()

        # Host-side compose (keep native bfloat16 — skip float32 conversion)
        result = ttnn.to_torch(
            self.host_staging,
            mesh_composer=composer,
        )
        t_end = time.perf_counter()

        self.last_timing["d2h_ms"] = (t_end - t0) * 1000
        self.last_timing["staging_wait_ms"] = (t_wait - t0) * 1000
        self.last_timing["pcie_d2h_ms"] = (t_pcie - t_wait) * 1000
        self.last_timing["compose_ms"] = (t_end - t_pcie) * 1000
        return result

    def pcie_d2h(self, async_d2h=False, return_slot=False):
        """PCIe D2H only — does NOT compose.

        Returns False on the very first call (no previous frame yet).
        Otherwise returns True (default) or the slot id that was just
        written into when ``return_slot=True`` — pass that slot to
        ``compose(slot=...)`` so the BG thread reads the right buffer
        even if a later pcie_d2h call has moved the ring index on.

        If ``async_d2h`` is True, enqueues the D2H on CQ1 and returns
        immediately; the caller (or ``compose()``) must call
        ``ttnn.event_synchronize(read_done_event)`` before reading
        ``host_staging``.  This lets the host pipeline submit(N+1)
        during d2h(N-1).

        If ``async_d2h`` is False (default), blocks until the PCIe
        transfer completes — preserves prior behavior.
        """
        if self._pipeline_frame <= 1:
            return False

        t0 = time.perf_counter()
        # Read the most recently written ring slot (the slot enqueue_frame
        # just wrote in this iteration's submit).
        slot = (self._stg_write_idx - 1) % self.staging_ring
        ds = self.dram_staging_ring[slot]
        hs = self.host_staging_ring[slot]
        ttnn.wait_for_event(1, self.staging_ready_event_ring[slot])
        t_wait = time.perf_counter()

        ttnn.copy_device_to_host_tensor(ds, hs, blocking=not async_d2h, cq_id=1)
        self.read_done_event_ring[slot] = ttnn.record_event(self.device, 1)
        # Maintain the legacy single-buffer aliases for the compose path /
        # external callers that read .host_staging or .read_done_event.
        self.dram_staging = ds
        self.host_staging = hs
        self.read_done_event = self.read_done_event_ring[slot]
        self._last_read_idx = slot
        t_pcie = time.perf_counter()

        self.last_timing["staging_wait_ms"] = (t_wait - t0) * 1000
        self.last_timing["pcie_d2h_ms"] = (t_pcie - t_wait) * 1000
        return slot if return_slot else True

    def compose(self, mesh_composer=None, dest=None, wait_d2h=False, slot=None):
        """Host-side compose of previously D2H'd data.

        Must be called after ``pcie_d2h()``.

        Uses ``batch_to_torch`` — a single C++ call that memcpy's all 32
        shard buffers contiguously into a pre-allocated torch tensor.
        Eliminates per-shard Python to_torch calls, per-shard allocations,
        and torch.cat entirely.

        If ``dest`` is provided, writes directly into it; otherwise uses
        the pre-allocated ``_compose_buf``.

        Set ``wait_d2h=True`` when ``pcie_d2h`` was called with
        ``async_d2h=True``; this thread will event-sync on
        ``read_done_event`` before reading ``host_staging`` so the BG
        compose can be launched immediately after the non-blocking d2h.

        Thread-safe: only writes to ``_compose_timing`` (not ``last_timing``
        which may be overwritten by a concurrent ``submit`` call).
        """
        t0 = time.perf_counter()
        # The caller passes ``slot`` (returned by ``pcie_d2h``) so the
        # background compose thread reads the buffer that pcie_d2h wrote,
        # even after subsequent pcie_d2h calls have moved _last_read_idx on.
        if slot is None:
            slot = self._last_read_idx if self._last_read_idx >= 0 else 0
        slot = slot % self.staging_ring
        hs = self.host_staging_ring[slot]
        ev = self.read_done_event_ring[slot]
        if wait_d2h:
            ttnn.event_synchronize(ev)
        t_sync = time.perf_counter()
        out = dest if dest is not None else self._compose_buf
        hs.batch_to_torch(out, physical=self._compose_physical)
        t_end = time.perf_counter()

        self._compose_timing = (t_end - t0) * 1000
        self._compose_d2h_wait_ms = (t_sync - t0) * 1000
        return out

    def compose_into(self, dest):
        """Compose directly into a pre-allocated tensor, skipping torch.cat.

        Uses ``get_device_tensors`` + per-shard ``to_torch`` to avoid the
        single monolithic ``torch.cat`` that ``ConcatMeshToTensor`` performs.
        Copies each shard in-place to ``dest[i]`` (works with shared memory).
        ``dest`` must have shape ``[N, 84, 8400]`` where N >= num_devices.

        Must be called after ``pcie_d2h()``.
        """
        t0 = time.perf_counter()
        shards = ttnn.get_device_tensors(self.host_staging)
        t_extract = time.perf_counter()
        for i, shard in enumerate(shards):
            dest[i].copy_(shard.to_torch()[0])  # [1, 84, 8400] → [84, 8400], in-place
        t_end = time.perf_counter()
        self.last_timing["compose_ms"] = (t_end - t0) * 1000
        self.last_timing["compose_extract_ms"] = (t_extract - t0) * 1000
        self.last_timing["compose_copy_ms"] = (t_end - t_extract) * 1000
        self.last_timing["d2h_ms"] = (
            self.last_timing["staging_wait_ms"] + self.last_timing["pcie_d2h_ms"] + self.last_timing["compose_ms"]
        )
        return dest

    def flush_pipeline(self, mesh_composer=None):
        """Get the last frame's result after the final ``submit`` call.

        Syncs the device and reads the output directly (no staging copy needed
        since no concurrent CQ0 execution after sync).
        """
        ttnn.synchronize_device(self.device)
        composer = mesh_composer or self.mesh_composer
        return ttnn.to_torch(
            self.runner_infra.output_tensor[0],
            mesh_composer=composer,
            device=self.device,
        )

    def release(self):
        ttnn.release_trace(self.device, self.tid)
