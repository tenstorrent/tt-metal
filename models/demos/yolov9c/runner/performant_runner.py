# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov9c.runner.performant_runner_infra import YOLOv9PerformanceRunnerInfra


class YOLOv9PerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=1,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        model_task="segment",
        model_location_generator=None,
        resolution=(640, 640),
        torch_input_tensor=None,
        mesh_mapper=None,
        weights_mesh_mapper=None,
        mesh_composer=None,
    ):
        self.inputs_mesh_mapper = mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = mesh_composer
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor

        self.runner_infra = YOLOv9PerformanceRunnerInfra(
            device,
            device_batch_size,
            act_dtype,
            weight_dtype,
            model_task,
            model_location_generator,
            resolution=resolution,
            torch_input_tensor=self.torch_input_tensor,
            mesh_mapper=self.inputs_mesh_mapper,
            weights_mesh_mapper=self.weights_mesh_mapper,
            mesh_composer=self.outputs_mesh_composer,
        )

        self.device = device
        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self._capture_yolov9_trace_2cqs()

    def _prepare_input_for_compute(self):
        """
        Transfers input from host to device using CQ1,
        synchronizes with CQ0, and sets up input tensor.
        """
        # CQ1: transfer input tensor
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)

        # CQ0: wait for input transfer to finish
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)

    def _capture_yolov9_trace_2cqs(self):
        # Initial op event
        self.op_event = ttnn.record_event(self.device, 0)

        # === First Run: Setup and JIT ===
        self._prepare_input_for_compute()
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # === Optimized Run ===
        self._prepare_input_for_compute()
        self.runner_infra.run()
        self.runner_infra.validate()

        # === Capture Trace ===
        self._prepare_input_for_compute()
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()

        # Begin trace capture on compute queue
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(self.runner_infra.input_tensor.spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

        assert trace_input_addr == self.input_tensor.buffer_address()

    def _execute_yolov9_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

        return self.runner_infra.output_tensor

    def run(self, torch_input_tensor, check_pcc=False):
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        output = self._execute_yolov9_trace_2cqs_inference(tt_inputs_host=tt_inputs_host)
        if check_pcc:
            torch_output = self.runner_infra.torch_model(torch_input_tensor)
            self.runner_infra.validate(output, torch_output)

        return output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
