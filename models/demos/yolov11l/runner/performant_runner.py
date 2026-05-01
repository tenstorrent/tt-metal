# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

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
    ):
        # Periodic device-compute timing (Option 4 — same pattern as
        # yolov8l/runner/performant_runner.py:392-402). Wraps execute_trace
        # with two record_event calls and a paired event_synchronize so
        # the diff measures pure on-chip trace runtime, free of host
        # h2d/sync overhead. Critical: events MUST be recorded directly
        # adjacent to execute_trace (no host code between them) — if any
        # host work runs between the events, the device may finish the
        # trace before we sync, and the diff measures Python overhead
        # (yielding bogus near-zero compute_ms).
        self._compute_measure_every = 100
        self._pipeline_frame = 0
        self._last_compute_ms: float | None = None
        self.device = device
        self.resolution = resolution
        self.torch_input_tensor = torch_input_tensor

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
        self._capture_yolov11_trace_2cqs()

    def _capture_yolov11_trace_2cqs(self):
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
        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        # Capture
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
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        assert trace_input_addr == self.input_tensor.buffer_address()

    def _execute_yolov11_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        # Option-4 device-compute measurement, mirroring
        # yolov8l/runner/performant_runner.py:392-402 verbatim. The start
        # event is recorded LITERALLY right before execute_trace and the
        # end event LITERALLY right after — zero host work between them
        # — so the device queue is guaranteed to still be processing the
        # trace when we event_synchronize. Diff = pure on-chip trace time.
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

    def run(self, torch_input_tensor=None, check_pcc=False):
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        output = self._execute_yolov11_trace_2cqs_inference(tt_inputs_host)
        if check_pcc:
            self._validate(torch_input_tensor, output)

        return output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
