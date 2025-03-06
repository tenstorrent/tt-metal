# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import (
    is_wormhole_b0,
)
from models.demos.yolov4.tests.yolov4_test_infra import create_test_infra

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def buffer_address(tensor):
    return tensor.buffer_address()


# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


class Yolov4Trace2CQ:
    def __init__(self):
        ...

    def initialize_yolov4_trace_2cqs_inference(
        self,
        device,
        device_batch_size=1,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        model_location_generator=None,
    ):
        self.test_infra = create_test_infra(
            device,
            device_batch_size,
            act_dtype,
            weight_dtype,
            model_location_generator=model_location_generator,
        )
        self.tt_inputs_host, sharded_mem_config_DRAM, self.input_mem_config = self.test_infra.setup_dram_sharded_input(
            device
        )
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)

        # Initialize the op event so we can write
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
        self.test_infra.dealloc_output()

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
        self.test_infra.dealloc_output()
        trace_input_addr = ttnn.buffer_address(self.test_infra.input_tensor)
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.test_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        assert trace_input_addr == ttnn.buffer_address(self.input_tensor)

        self.device = device

    def get_region_boxes(self, boxes_and_confs):
        boxes_list = []
        confs_list = []

        for item in boxes_and_confs:
            boxes_list.append(item[0])
            confs_list.append(item[1])

        boxes = torch.cat(boxes_list, dim=1)
        confs = torch.cat(confs_list, dim=1)

        return [boxes, confs]

    def execute_yolov4_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)

        ttnn_output_tensor = self.test_infra.output_tensor

        result_boxes_padded = ttnn.to_torch(ttnn_output_tensor[0])
        result_confs = ttnn.to_torch(ttnn_output_tensor[1])

        result_boxes_padded = result_boxes_padded.permute(0, 2, 1, 3)
        result_boxes_list = []
        # That ttnn tensor is the concat output of 3 padded tensors
        # As a perf workaround I'm doing the unpadding on the torch output here.
        # TODO: cleaner ttnn code when ttnn.untilize() is fully optimized
        box_1_start_i = 0
        box_1_end_i = 6100
        box_2_start_i = 6128
        box_2_end_i = 6228
        box_3_start_i = 6256
        box_3_end_i = 6356
        result_boxes_list.append(result_boxes_padded[:, box_1_start_i:box_1_end_i])
        result_boxes_list.append(result_boxes_padded[:, box_2_start_i:box_2_end_i])
        result_boxes_list.append(result_boxes_padded[:, box_3_start_i:box_3_end_i])
        result_boxes = torch.cat(result_boxes_list, dim=1)

        return [result_boxes, result_confs]

    def release_yolov4_trace_2cqs_inference(self):
        ttnn.release_trace(self.device, self.tid)

    def run_traced_inference(self, torch_input_tensor):
        n, h, w, c = torch_input_tensor.shape
        torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
        return self.execute_yolov4_trace_2cqs_inference(tt_inputs_host)
