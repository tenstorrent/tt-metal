# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.mobileNetV3.runner.performant_runner_infra import MobileNetV3PerformanceRunnerInfra


class MobileNetV3PerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=1,
        model_location_generator=None,
        resolution=(224, 224),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        input_path=None,
    ):
        self.device = device
        self.resolution = resolution
        self.torch_input_tensor = torch_input_tensor

        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer

        self.runner_infra = MobileNetV3PerformanceRunnerInfra(
            device,
            device_batch_size,
            model_location_generator,
            resolution=resolution,
            torch_input_tensor=self.torch_input_tensor,
            inputs_mesh_mapper=self.inputs_mesh_mapper,
            weights_mesh_mapper=self.weights_mesh_mapper,
            outputs_mesh_composer=self.outputs_mesh_composer,
            input_path=input_path,
        )

        # Inputs to MobileNetV3 need to be in ttnn.DRAM_MEMORY_CONFIG for supporting DRAM sliced Conv2d
        (
            self.tt_inputs_host,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_interleaved_input()
        self._capture_MobileNetV3_trace_2cqs()

    def _capture_MobileNetV3_trace_2cqs(self):
        # Setting up persistent DRAM buffer for input tensor
        self.input_dram_tensor = ttnn.allocate_tensor_on_device(
            self.tt_inputs_host.shape,
            self.tt_inputs_host.dtype,
            self.tt_inputs_host.layout,
            self.device,
            self.input_mem_config,
        )

        # Initialize the op event so we can write
        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.input_dram_tensor
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

    def _execute_MobileNetV3_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.input_dram_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.runner_infra.tt_output

    def _validate(self):
        self._PCC_THRESH = 0.98
        checks = [
            ("MobileNetV3", ttnn.to_torch(self.tt_output), self.runner_infra.torch_output),
        ]
        for name, tt_out, torch_ref in checks:
            assert_with_pcc(torch_ref, tt_out, pcc=self._PCC_THRESH)

    def run(self, torch_input_tensor=None, check_pcc=False):
        torch_input_tensor = self.runner_infra.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        # Converting to NHWC format for TTNN operations
        if torch_input_tensor.shape[-1] != 3:
            torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        (
            self.tt_inputs_host,
            _,
        ) = self.runner_infra.setup_dram_interleaved_input()

        # Resetting output tensors for output validity checking
        self.tt_output = None

        self.tt_output = self._execute_MobileNetV3_trace_2cqs_inference(tt_inputs_host=self.tt_inputs_host)

        # Validate outputs from trace
        if check_pcc:
            self._validate()
        return self.tt_output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
