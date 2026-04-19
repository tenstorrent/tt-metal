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

        # Capture
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
        self.input_tensor = ttnn.allocate_tensor_on_device(dram_spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        # assert trace_input_addr == self.input_tensor.buffer_address()

    def _setup_staging_buffer(self):
        """Create a staging buffer for pipelined D2H.

        Uses ``ttnn.clone`` to create a copy at a separate DRAM address with
        the same layout/memory-config as the model output.  During pipelined
        execution, CQ1 reads from staging while CQ0 writes to the trace's
        fixed output address — no conflict because they're at different addrs.
        """
        output = self.runner_infra.output_tensor[0]
        # Clone preserves layout (TILE) and memory config (interleaved DRAM)
        self.dram_staging = ttnn.clone(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Events: staging_ready (CQ0: staging written), read_done (CQ1: D2H finished)
        self.staging_ready_event = ttnn.record_event(self.device, 0)
        self.read_done_event = ttnn.record_event(self.device, 1)
        self._pipeline_frame = 0

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
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)

        # CQ0: wait for H2D, then copy prev output → staging, then compute
        ttnn.wait_for_event(0, self.write_event)

        # Copy previous output to staging (safe: staging not read by CQ1 yet)
        if self._pipeline_frame > 0:
            ttnn.wait_for_event(0, self.read_done_event)  # prev D2H released staging
            ttnn.deallocate(self.dram_staging)
            self.dram_staging = ttnn.clone(
                self.runner_infra.output_tensor[0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.staging_ready_event = ttnn.record_event(self.device, 0)

        # Reshard input + start trace (all on CQ0, after reshard-to-staging)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        t_queue = (time.perf_counter() - t0) * 1000
        self._pipeline_frame += 1
        self.last_timing = {
            "host_prep_ms": self._last_host_prep_ms,
            "queue_ms": t_queue,
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
        result = ttnn.to_torch(
            self.dram_staging,
            dtype=torch.float32,
            mesh_composer=composer,
            device=self.device,
            cq_id=1,
        )
        self.read_done_event = ttnn.record_event(self.device, 1)
        self.last_timing["d2h_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def flush_pipeline(self, mesh_composer=None):
        """Get the last frame's result after the final ``submit`` call.

        Syncs the device, reshards the final output to staging, and D2Hs.
        """
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(self.dram_staging)
        self.dram_staging = ttnn.clone(
            self.runner_infra.output_tensor[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        composer = mesh_composer or self.mesh_composer
        return ttnn.to_torch(
            self.dram_staging,
            dtype=torch.float32,
            mesh_composer=composer,
            device=self.device,
        )

    def release(self):
        ttnn.release_trace(self.device, self.tid)
