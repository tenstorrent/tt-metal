# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import ttnn
from models.demos.yolov8s.runner.performant_runner_infra import YOLOv8sPerformanceRunnerInfra

try:
    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class YOLOv8sPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        mesh_mapper=None,
        mesh_composer=None,
        weights_mesh_mapper=None,
        model_location_generator=None,
    ):
        # Periodic device-compute measurement (Option 4). Same convention as
        # the v8l runner — see _execute_yolov8s_trace_2cqs_inference for the
        # measurement block.
        self._compute_measure_every = 100
        self._pipeline_frame = 0
        self._last_compute_ms: float | None = None
        self.device = device
        self.mesh_mapper = mesh_mapper
        self.mesh_composer = mesh_composer
        self.weights_mesh_mapper = weights_mesh_mapper
        self.runner_infra = YOLOv8sPerformanceRunnerInfra(
            device,
            device_batch_size,
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
        self._capture_yolov8s_trace_2cqs()

    def _capture_yolov8s_trace_2cqs(self):
        # Initialize the op event so we can write
        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT
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

    def _execute_yolov8s_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)

        # Option-4 device-compute measurement (mirrors yolov8l/runner/
        # performant_runner.py:392-402). Start event recorded literally
        # right before execute_trace, end event right after — no host
        # work between them — so the diff is pure on-chip trace runtime.
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

    def run(self, torch_input_tensor=None):
        if torch_input_tensor is None:
            return self._execute_yolov8s_trace_2cqs_inference()
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        return self._execute_yolov8s_trace_2cqs_inference(tt_inputs_host)

    def release(self):
        ttnn.release_trace(self.device, self.tid)
