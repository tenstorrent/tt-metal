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
    ):
        self.device = device
        self.mesh_mapper = mesh_mapper
        self.mesh_composer = mesh_composer
        self.weights_mesh_mapper = weights_mesh_mapper
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
        ) = self.runner_infra._setup_dram_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self._capture_yolov8l_trace_2cqs()

    def _convert_tensor_to_input_config(self, tensor):
        """Convert tensor to the appropriate memory configuration for input."""
        # Keep original DRAM-sharded format during trace capture
        return tensor

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
        self.runner_infra.validate()

        # Untilize detection output on-device (TILE → ROW_MAJOR).
        # This eliminates the expensive CPU-side untiling in to_torch,
        # cutting compose from ~18ms to ~10ms.
        tile_output = self.runner_infra.output_tensor[0]
        rm_output = ttnn.untilize(tile_output, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tile_output)
        self.runner_infra.output_tensor[0] = rm_output

        # Pre-allocate staging buffer for pipelined D2H/compute overlap.
        # Must happen BEFORE trace capture so the trace allocator avoids
        # this address — post-trace allocations get corrupted when trace
        # execution overwrites intermediate buffers at reused addresses.
        self.dram_staging = ttnn.clone(
            self.runner_infra.output_tensor[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Pre-allocate host buffer — avoids per-frame allocation overhead
        self.host_staging = ttnn.allocate_tensor_on_host(self.dram_staging.spec, self.device)
        self.staging_ready_event = ttnn.record_event(self.device, 0)
        self.read_done_event = ttnn.record_event(self.device, 1)
        self._pipeline_frame = 0
        self._compose_timing = 0.0

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
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        # On-device output processing inside trace — replayed every frame
        tile_output_traced = self.runner_infra.output_tensor[0]
        self.runner_infra.output_tensor[0] = ttnn.untilize(
            tile_output_traced, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(tile_output_traced)
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
        # TODO: Add in place support to ttnn to_memory_config
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)

        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        return self.runner_infra.output_tensor

    # ------------------------------------------------------------------
    # Original synchronous API (unchanged)
    # ------------------------------------------------------------------

    def run(self, torch_input_tensor):
        t0 = time.perf_counter()
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
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
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        self._last_host_prep_ms = (time.perf_counter() - t0) * 1000
        return tt_inputs_host

    def enqueue_frame(self, tt_inputs_host=None):
        """Queue H2D + staging copy + reshard + trace.  Non-blocking.

        If ``tt_inputs_host`` is None, uses the last prepared input.
        Must be called after ``prepare_input`` (or pass the result directly).
        """
        t0 = time.perf_counter()
        # Queue H2D on CQ1
        ttnn.wait_for_event(1, self.op_event)
        t1 = time.perf_counter()
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        t2 = time.perf_counter()
        self.write_event = ttnn.record_event(self.device, 1)

        # CQ0: wait for H2D, then copy prev output → staging, then compute
        ttnn.wait_for_event(0, self.write_event)
        t3 = time.perf_counter()

        # Copy previous output to staging (safe: staging not read by CQ1 yet)
        if self._pipeline_frame > 0:
            ttnn.wait_for_event(0, self.read_done_event)  # prev D2H released staging
            ttnn.copy(self.runner_infra.output_tensor[0], self.dram_staging)
            self.staging_ready_event = ttnn.record_event(self.device, 0)
        t4 = time.perf_counter()

        # Reshard input + start trace (all on CQ0, after reshard-to-staging)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        t5 = time.perf_counter()
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

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

    def pcie_d2h(self):
        """PCIe D2H only — does NOT compose.

        Blocks until PCIe transfer completes.  Returns False on the first
        call (no previous frame yet).  Call ``compose()`` afterwards to
        get the torch result.
        """
        if self._pipeline_frame <= 1:
            return False

        t0 = time.perf_counter()
        ttnn.wait_for_event(1, self.staging_ready_event)
        t_wait = time.perf_counter()

        ttnn.copy_device_to_host_tensor(self.dram_staging, self.host_staging, blocking=True, cq_id=1)
        self.read_done_event = ttnn.record_event(self.device, 1)
        t_pcie = time.perf_counter()

        self.last_timing["staging_wait_ms"] = (t_wait - t0) * 1000
        self.last_timing["pcie_d2h_ms"] = (t_pcie - t_wait) * 1000
        return True

    def compose(self, mesh_composer=None, dest=None):
        """Host-side compose of previously D2H'd data.

        Must be called after ``pcie_d2h()``.

        Uses ``batch_to_torch`` — a single C++ call that memcpy's all 24
        shard buffers contiguously into a pre-allocated torch tensor.
        Eliminates per-shard Python to_torch calls, per-shard allocations,
        and torch.cat entirely.

        If ``dest`` is provided, writes directly into it; otherwise uses
        the pre-allocated ``_compose_buf``.

        Thread-safe: only writes to ``_compose_timing`` (not ``last_timing``
        which may be overwritten by a concurrent ``submit`` call).
        """
        t0 = time.perf_counter()
        out = dest if dest is not None else self._compose_buf
        self.host_staging.batch_to_torch(out, physical=self._compose_physical)
        t_end = time.perf_counter()

        self._compose_timing = (t_end - t0) * 1000
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
