# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.retinanet.runner.performant_runner_infra import RetinaNetPerformanceRunnerInfra


class RetinaNetPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=1,
        model_location_generator=None,
        resolution=(512, 512),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        input_path=None,
        model_config=None,
    ):
        self.device = device
        self.resolution = resolution
        self.torch_input_tensor = torch_input_tensor
        self.num_devices = self.device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer

        # Use default model_config if not provided
        if model_config is None:
            model_config = {
                "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
                "WEIGHTS_DTYPE": ttnn.bfloat16,
                "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            }

        self.runner_infra = RetinaNetPerformanceRunnerInfra(
            device,
            device_batch_size,
            model_config,
            model_location_generator,
            resolution=resolution,
            torch_input_tensor=self.torch_input_tensor,
            inputs_mesh_mapper=self.inputs_mesh_mapper,
            weights_mesh_mapper=self.weights_mesh_mapper,
            outputs_mesh_composer=self.outputs_mesh_composer,
            input_path=input_path,
        )

        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_sharded_input(device)

        # Setup DRAM interleaved input (not sharded)
        self.tt_inputs_host, self.input_mem_config = self.runner_infra.setup_dram_interleaved_input()
        # Move to device with DRAM interleaved memory
        self.tt_image_res = ttnn.to_device(self.tt_inputs_host, device, self.input_mem_config)

        self._capture_retinanet_trace_2cqs()

    def _capture_retinanet_trace_2cqs(self):
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

    def _execute_retinanet_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return {
            "regression": self.runner_infra.regression_output,
            "classification": self.runner_infra.classification_output,
        }

    def _validate(self):
        self.runner_infra.validate()

    def run(self, torch_input_tensor, check_pcc=False):
        n, c, h, w = torch_input_tensor.shape
        tt_inputs_host, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
        output = self._execute_retinanet_trace_2cqs_inference(tt_inputs_host)
        if check_pcc:
            self._validate()
        return output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
