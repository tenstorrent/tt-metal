# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11s.runner.performant_runner_infra import YOLOv11sPerformanceRunnerInfra
from tests.ttnn.utils_for_testing import assert_with_pcc


class YOLOv11sPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=1,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        model_location_generator=None,
        resolution=(640, 640),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
    ):
        self.device = device
        self.resolution = resolution
        self.torch_input_tensor = torch_input_tensor

        self.mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.mesh_composer = outputs_mesh_composer

        self.runner_infra = YOLOv11sPerformanceRunnerInfra(
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
        self.tt_image_res = [
            self.tt_inputs_host.to(device, sharded_mem_config_DRAM),
            self.tt_inputs_host.to(device, sharded_mem_config_DRAM),
        ]
        self._dram_ping = 0
        self._perf_tt_inputs = None
        self._perf_torch_ref = None
        self._capture_yolov11s_trace_2cqs()

    def _capture_yolov11s_trace_2cqs(self):
        def dram_buf():
            return self.tt_image_res[self._dram_ping]

        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, dram_buf(), 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(dram_buf(), self.input_mem_config)
        self._dram_ping ^= 1
        spec = self.runner_infra.input_tensor.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, dram_buf(), 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(dram_buf(), self.input_mem_config)
        self._dram_ping ^= 1
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, dram_buf(), 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(dram_buf(), self.input_mem_config)
        self._dram_ping ^= 1
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        assert trace_input_addr == self.input_tensor.buffer_address()
        self._dram_ping = 0

    def _execute_yolov11s_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        dram = self.tt_image_res[self._dram_ping]
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, dram, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(dram, self.input_mem_config, self.input_tensor)
        self._dram_ping ^= 1
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.runner_infra.output_tensor

    def _validate(self, input_tensor, result_output_tensor):
        torch_output_tensor = self.runner_infra.torch_output_tensor
        assert_with_pcc(torch_output_tensor, result_output_tensor, 0.99)

    def run(self, torch_input_tensor=None, tt_inputs_host=None, check_pcc=False):
        if tt_inputs_host is not None:
            host_in = tt_inputs_host
        elif torch_input_tensor is not None:
            if self._perf_torch_ref is not torch_input_tensor:
                self._perf_torch_ref = torch_input_tensor
                self._perf_tt_inputs, _ = self.runner_infra._setup_l1_sharded_input(self.device, torch_input_tensor)
            host_in = self._perf_tt_inputs
        else:
            host_in = self.tt_inputs_host
        output = self._execute_yolov11s_trace_2cqs_inference(host_in)
        if check_pcc:
            self._validate(torch_input_tensor, output)

        return output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
