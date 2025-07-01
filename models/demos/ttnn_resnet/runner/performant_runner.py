# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.ttnn_resnet.runner.performant_runner_infra import ResNet50PerformanceRunnerInfra


class ResNet50PerformantRunner:
    def __init__(self, device, device_batch_size=1, act_dtype=ttnn.bfloat16, weight_dtype=ttnn.bfloat16):
        self.runner_infra = ResNet50PerformanceRunnerInfra(
            device,
            device_batch_size,
            act_dtype,
            weight_dtype,
            ttnn.MathFidelity.LoFi,
            True,
            dealloc_input=True,
            final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.device = device
        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
        ) = self.runner_infra.setup_dram_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self.capture_resnet50_trace_2cqs()

    def capture_resnet50_trace_2cqs(self):
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
        self.runner_infra.output_tensor.deallocate(force=True)

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
        self.runner_infra.output_tensor.deallocate(force=True)
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        assert trace_input_addr == self.input_tensor.buffer_address()

    def execute_resnet50_trace_2cqs_inference(self, tt_inputs_host=None):
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        outputs = ttnn.from_device(self.runner_infra.output_tensor, blocking=True)

        return outputs

    def run(self, torch_input_tensor, check_pcc=False):
        n, c, h, w = torch_input_tensor.shape
        tt_inputs_host, input_mem_config = self.runner_infra.setup_l1_sharded_input(self.device, torch_input_tensor)
        output = self.execute_resnet50_trace_2cqs_inference(tt_inputs_host)

        if check_pcc:
            self.validate(torch_input_tensor, output)

        return output

    def release_resnet50_trace_2cqs_inference(self):
        ttnn.release_trace(self.device, self.tid)
