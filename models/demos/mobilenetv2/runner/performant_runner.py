# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.mobilenetv2.runner.performant_runner_infra import create_test_infra

try:
    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class MobileNetV2Trace2CQ:
    def __init__(self):
        ...

    def initialize_mobilenetv2_trace_2cqs_inference(
        self,
        device,
        device_batch_size,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
    ):
        self.test_infra = create_test_infra(
            device,
            device_batch_size,
        )
        self.device = device
        self.tt_inputs_host, sharded_mem_config_DRAM, self.input_mem_config = self.test_infra.setup_dram_sharded_input(
            device
        )
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self.op_event = ttnn.record_event(device, 0)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.test_infra.input_tensor.spec
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.run()
        self.test_infra.validate()
        self.test_infra.output_tensor.deallocate(force=True)

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.run()
        self.test_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.output_tensor.deallocate(force=True)
        trace_input_addr = self.test_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.test_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        assert trace_input_addr == self.input_tensor.buffer_address()

    def execute_mobilenetv2_trace_2cqs_inference(self, tt_inputs_host=None):
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        outputs = ttnn.from_device(self.test_infra.output_tensor, blocking=True)
        return outputs

    def release_mobilenetv2_trace_2cqs_inference(self):
        ttnn.release_trace(self.device, self.tid)
