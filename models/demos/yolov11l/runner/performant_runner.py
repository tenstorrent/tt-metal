# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import torch

import ttnn
from models.demos.yolov11l.runner.performant_runner_infra import YOLOv11PerformanceRunnerInfra
from tests.ttnn.utils_for_testing import assert_with_pcc


class YOLOv11PerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=1,  # total mesh input batch N (= shard count); same convention as YOLOv8l (not per-chip × mesh again).
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        model_location_generator=None,
        resolution=(640, 640),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        compact_output=False,
        staging_ring=1,
    ):
        # Periodic device-compute timing (Option 4) — events around execute_trace.
        self._compute_measure_every = 100
        self._pipeline_frame = 0
        self._last_compute_ms: float | None = None
        self._last_host_prep_ms: float = 0.0

        self.device = device
        self.resolution = resolution
        self.torch_input_tensor = torch_input_tensor
        self.compact_output = compact_output
        self.staging_ring = max(1, int(staging_ring))

        self.mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.mesh_composer = outputs_mesh_composer

        self.runner_infra = YOLOv11PerformanceRunnerInfra(
            device,
            device_batch_size,
            act_dtype,
            weight_dtype,
            model_location_generator,
            resolution=resolution,
            torch_input_tensor=self.torch_input_tensor,
            inputs_mesh_mapper=self.mesh_mapper,
            weights_mesh_mapper=self.weights_mesh_mapper,
            outputs_mesh_composer=self.mesh_composer,
        )

        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self.last_timing = {}
        self._capture_yolov11_trace_2cqs()

    # ------------------------------------------------------------------
    # Compact-output: collapse 80-class YOLO output to 1 max-conf + 1 argmax
    # channel.  Same idea as yolov8l/runner/performant_runner.py:_reduce_classes,
    # adapted to YOLOv11l's [1, 84, 8400] output shape (84 = 4 bbox + 80 classes).
    # ------------------------------------------------------------------
    def _reduce_classes(self, rm_output):
        """[B, 84, A] → [B, 6, A] via on-device class reduction.

        Channels 0..3 = box (cx, cy, w, h) preserved; channels 4..83 (80 COCO classes)
        collapsed to channel 4 = max conf and channel 5 = argmax class id.

        Cuts D2H volume 84/6 ≈ 14× — the dominant per-frame cost in the demo pipeline.
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

    def _capture_yolov11_trace_2cqs(self):
        # First run: configures convs JIT (not in trace)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.runner_infra.input_tensor.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run + figure out output shape (tile padding) + apply compact_output
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        dram_spec = self.runner_infra.input_tensor.spec  # For trace
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        if not self.compact_output:
            self.runner_infra.validate()

        # Yolov11l's output is a single tensor (not a list like v8l). Unlike v8l,
        # v11l's TtnnDetect already returns ROW_MAJOR — skip untilize in that case.
        tile_output = self.runner_infra.output_tensor
        _logical = list(tile_output.shape)
        _padded = list(tile_output.padded_shape)
        self._is_tile_output = tile_output.layout == ttnn.TILE_LAYOUT
        self._needs_unpad = self._is_tile_output and _padded != _logical
        if self._is_tile_output:
            if self._needs_unpad:
                self._output_end = tuple(s - 1 for s in _logical)
                rm_output = ttnn.untilize_with_unpadding(
                    tile_output,
                    output_tensor_end=self._output_end,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                print(
                    f"[v11l runner] untilize_with_unpadding: {_padded} → {_logical} "
                    f"(saves {(_padded[-2]-_logical[-2])*_padded[-1]*2/1024:.0f} KB/shard)",
                    flush=True,
                )
            else:
                rm_output = ttnn.untilize(tile_output, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(tile_output)
        else:
            # Already ROW_MAJOR; pass through (clone to DRAM so the trace can replace it).
            rm_output = ttnn.clone(tile_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(tile_output)
        if self.compact_output:
            compact_rm = self._reduce_classes(rm_output)
            ttnn.deallocate(rm_output)
            self.runner_infra.output_tensor = compact_rm
            print(
                f"[v11l runner] compact_output: 84-channel → 6-channel "
                f"(box[4] + max_conf[1] + argmax_id[1]) — ~14× D2H reduction",
                flush=True,
            )
        else:
            self.runner_infra.output_tensor = rm_output

        # Pre-allocate K staging buffer pairs for pipelined D2H/compute overlap.
        # Must happen BEFORE trace capture so the trace allocator avoids these.
        K = self.staging_ring
        self.dram_staging_ring = []
        self.host_staging_ring = []
        self.staging_ready_event_ring = []
        self.read_done_event_ring = []
        for _ in range(K):
            ds = ttnn.clone(self.runner_infra.output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hs = ttnn.allocate_tensor_on_host(ds.spec, self.device)
            self.dram_staging_ring.append(ds)
            self.host_staging_ring.append(hs)
            # Initial events are "already done" so first writer doesn't block.
            self.staging_ready_event_ring.append(ttnn.record_event(self.device, 0))
            self.read_done_event_ring.append(ttnn.record_event(self.device, 1))
        # Backwards-compatible single-buffer aliases (slot 0).
        self.dram_staging = self.dram_staging_ring[0]
        self.host_staging = self.host_staging_ring[0]
        self.staging_ready_event = self.staging_ready_event_ring[0]
        self.read_done_event = self.read_done_event_ring[0]
        self._stg_write_idx = 0
        self._last_read_idx = -1
        self._compose_timing = 0.0
        if K > 1:
            print(f"[v11l runner] staging_ring=K={K} (CQ0/CQ1 decoupled)", flush=True)

        # Pre-allocate compose output buffer sized for PHYSICAL shape (includes
        # tile-alignment padding). batch_to_torch(physical=True) copies the full
        # physical buffer contiguously. Callers slice to logical for use.
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
            f"[v11l runner] staging logical={list(_logical)} padded={list(_padded)} "
            f"physical=({_phys_h}×{_phys_w}) logical=({_log_h}×{_log_w}) "
            f"compose_physical={self._compose_physical} compose_buf={list(self._compose_buf.shape)}",
            flush=True,
        )

        # Capture trace
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        # On-device output processing inside trace — replayed every frame
        tile_output_traced = self.runner_infra.output_tensor
        if self._is_tile_output:
            if self._needs_unpad:
                rm_traced = ttnn.untilize_with_unpadding(
                    tile_output_traced,
                    output_tensor_end=self._output_end,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                rm_traced = ttnn.untilize(tile_output_traced, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(tile_output_traced)
        else:
            rm_traced = ttnn.clone(tile_output_traced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(tile_output_traced)
        if self.compact_output:
            compact_traced = self._reduce_classes(rm_traced)
            ttnn.deallocate(rm_traced)
            self.runner_infra.output_tensor = compact_traced
        else:
            self.runner_infra.output_tensor = rm_traced
        self.input_tensor = ttnn.allocate_tensor_on_device(dram_spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

    def _execute_yolov11_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        # Optional device-compute measurement (events around execute_trace).
        measure_compute = self._pipeline_frame % self._compute_measure_every == 0
        compute_start_evt = ttnn.record_event(self.device, 0) if measure_compute else None
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        if measure_compute:
            compute_end_evt = ttnn.record_event(self.device, 0)
            ttnn.event_synchronize(compute_start_evt)
            t_compute_start = time.perf_counter()
            ttnn.event_synchronize(compute_end_evt)
            self._last_compute_ms = (time.perf_counter() - t_compute_start) * 1000
        self._pipeline_frame += 1
        return self.runner_infra.output_tensor

    def _validate(self, input_tensor, result_output_tensor):
        torch_output_tensor = self.runner_infra.torch_output_tensor
        assert_with_pcc(torch_output_tensor, result_output_tensor, 0.99)

    # ------------------------------------------------------------------
    # Original synchronous API (kept for backward compat / perf tests)
    # ------------------------------------------------------------------
    def run(self, torch_input_tensor=None, check_pcc=False):
        t0 = time.perf_counter()
        if torch_input_tensor is None:
            tt_inputs_host = None
        else:
            tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        t1 = time.perf_counter()
        output = self._execute_yolov11_trace_2cqs_inference(tt_inputs_host)
        t2 = time.perf_counter()
        self.last_timing = {
            "host_prep_ms": (t1 - t0) * 1000,
            "h2d_and_trace_ms": (t2 - t1) * 1000,
        }
        if check_pcc:
            self._validate(torch_input_tensor, output)
        return output

    # ------------------------------------------------------------------
    # Pipelined API (mirrors yolov8l/runner/performant_runner.py)
    # ------------------------------------------------------------------

    def prepare_input(self, torch_input_tensor):
        """CPU-only host prep: torch → ttnn host (hugepage-backed). Returns tt_inputs_host."""
        t0 = time.perf_counter()
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        self._last_host_prep_ms = (time.perf_counter() - t0) * 1000
        return tt_inputs_host

    def enqueue_frame(self, tt_inputs_host=None):
        """H2D + staging copy + reshard + trace. Non-blocking."""
        t0 = time.perf_counter()
        ttnn.wait_for_event(1, self.op_event)
        t1 = time.perf_counter()
        if not getattr(self, "_skip_h2d", False):
            ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        t2 = time.perf_counter()
        self.write_event = ttnn.record_event(self.device, 1)

        ttnn.wait_for_event(0, self.write_event)
        t3 = time.perf_counter()

        # Copy previous frame's output to next staging-ring slot.
        if self._pipeline_frame > 0:
            slot = self._stg_write_idx % self.staging_ring
            ttnn.wait_for_event(0, self.read_done_event_ring[slot])
            ttnn.copy(self.runner_infra.output_tensor, self.dram_staging_ring[slot])
            self.staging_ready_event_ring[slot] = ttnn.record_event(self.device, 0)
            self.staging_ready_event = self.staging_ready_event_ring[slot]
            self._stg_write_idx += 1
        t4 = time.perf_counter()

        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        t5 = time.perf_counter()
        self.op_event = ttnn.record_event(self.device, 0)

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
        """prepare_input + enqueue_frame."""
        tt_inputs_host = self.prepare_input(torch_input_tensor)
        self.enqueue_frame(tt_inputs_host)

    def pcie_d2h(self, async_d2h=False, return_slot=False):
        """PCIe D2H only — does NOT compose. False on the first call."""
        if self._pipeline_frame <= 1:
            return False
        t0 = time.perf_counter()
        slot = (self._stg_write_idx - 1) % self.staging_ring
        ds = self.dram_staging_ring[slot]
        hs = self.host_staging_ring[slot]
        ttnn.wait_for_event(1, self.staging_ready_event_ring[slot])
        t_wait = time.perf_counter()
        ttnn.copy_device_to_host_tensor(ds, hs, blocking=not async_d2h, cq_id=1)
        self.read_done_event_ring[slot] = ttnn.record_event(self.device, 1)
        self.dram_staging = ds
        self.host_staging = hs
        self.read_done_event = self.read_done_event_ring[slot]
        self._last_read_idx = slot
        t_pcie = time.perf_counter()
        self.last_timing["staging_wait_ms"] = (t_wait - t0) * 1000
        self.last_timing["pcie_d2h_ms"] = (t_pcie - t_wait) * 1000
        return slot if return_slot else True

    def compose(self, mesh_composer=None, dest=None, wait_d2h=False, slot=None):
        """Host-side compose of previously D2H'd data. Call after pcie_d2h()."""
        t0 = time.perf_counter()
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

    def flush_pipeline(self, mesh_composer=None):
        """Drain remaining staging after final submit."""
        ttnn.synchronize_device(self.device)
        composer = mesh_composer or self.mesh_composer
        return ttnn.to_torch(self.runner_infra.output_tensor, mesh_composer=composer, device=self.device)

    def release(self):
        ttnn.release_trace(self.device, self.tid)
