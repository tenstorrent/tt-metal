# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.vit.tests.vit_test_infra import create_test_infra


class VitTrace2CQ:
    def __init__(self):
        ...

    def initialize_vit_trace_2cqs_inference(
        self,
        device,
        device_batch_size=8,
        use_random_input_tensor=False,
    ):
        self.test_infra = create_test_infra(
            device,
            device_batch_size,
            use_random_input_tensor=use_random_input_tensor,
        )
        self.device = device
        self.tt_inputs_host, sharded_mem_config_DRAM, self.input_mem_config = self.test_infra.setup_dram_sharded_input(
            device
        )
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)

        self.first_op_event = ttnn.record_event(device, 0)
        self.read_event = ttnn.record_event(device, 1)

        # JIT compilation
        ttnn.wait_for_event(1, self.first_op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.first_op_event = ttnn.record_event(device, 0)
        self.test_infra.run()
        self.output_tensor_dram = ttnn.to_memory_config(self.test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        self.last_op_event = ttnn.record_event(device, 0)

        # Capture trace
        ttnn.wait_for_event(1, self.first_op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.first_op_event = ttnn.record_event(device, 0)
        spec = self.test_infra.input_tensor.spec
        input_trace_addr = self.test_infra.input_tensor.buffer_address()

        self.test_infra.output_tensor.deallocate(force=True)
        self.trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        output_tensor = self.test_infra.run()

        self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, self.trace_id, cq_id=0)
        self.output_tensor_dram = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        assert input_trace_addr == self.input_tensor.buffer_address()

    def execute_vit_trace_2cqs_inference(self, tt_inputs_host=None, first_input=False):
        ttnn.wait_for_event(1, self.first_op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)

        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.first_op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, self.read_event)

        self.output_tensor_dram = ttnn.to_memory_config(self.test_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        self.last_op_event = ttnn.record_event(self.device, 0)

        ttnn.wait_for_event(1, self.last_op_event)
        outputs = ttnn.from_device(self.output_tensor_dram, blocking=True, cq_id=1)
        self.read_event = ttnn.record_event(self.device, 1)

        return outputs

    def release_vit_trace_2cqs_inference(self):
        ttnn.release_trace(self.device, self.trace_id)
