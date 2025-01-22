# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import (
    is_wormhole_b0,
)
from models.demos.yolov4.tests.yolov4_test_infra import create_test_infra
from models.demos.yolov4.demo.demo import YoloLayer


try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


def run_yolov4_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=model_location_generator,
    )

    tt_inputs_host, self.input_mem_config = test_infra.setup_l1_sharded_input(device)

    # # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device, self.input_mem_config)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, self.input_mem_config)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    test_infra.input_tensor = tt_inputs_host.to(device, self.input_mem_config)
    test_infra.run()
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()
    test_infra.dealloc_output()


def run_yolov4_trace_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, self.input_mem_config = test_infra.setup_l1_sharded_input(device)

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device, self.input_mem_config)
    spec = test_infra.input_tensor.spec
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, self.input_mem_config)
    test_infra.run()
    test_infra.validate()

    # Capture
    test_infra.input_tensor = tt_inputs_host.to(device, self.input_mem_config)
    test_infra.dealloc_output()
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    self.tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, self.tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(tt_image_res)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    ttnn.execute_trace(device, self.tid, cq_id=0, blocking=True)
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()

    ttnn.release_trace(device, self.tid)
    test_infra.dealloc_output()


def run_yolov4_trace_2cqs_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, sharded_mem_config_DRAM, self.input_mem_config = test_infra.setup_dram_sharded_input(device)
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)
    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    # First run configures convs JIT
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, self.input_mem_config)
    spec = test_infra.input_tensor.spec
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # Optimized run
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, self.input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, self.input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.dealloc_output()
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    self.tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, self.tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(self.input_tensor)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    for iter in range(0, 2):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        self.input_tensor = ttnn.reshard(tt_image_res, self.input_mem_config, self.input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, self.tid, cq_id=0, blocking=False)
    ttnn.synchronize_devices(device)

    if use_signpost:
        signpost(header="stop")

    ttnn.release_trace(device, self.tid)


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
        self.op_event = ttnn.create_event(device)
        self.write_event = ttnn.create_event(device)
        # Initialize the op event so we can write
        ttnn.record_event(0, self.op_event)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        ttnn.record_event(1, self.write_event)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.test_infra.input_tensor.spec
        ttnn.record_event(0, self.op_event)
        self.test_infra.run()
        self.test_infra.validate()
        self.test_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        ttnn.record_event(1, self.write_event)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        ttnn.record_event(0, self.op_event)
        self.test_infra.run()
        self.test_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        ttnn.record_event(1, self.write_event)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        ttnn.record_event(0, self.op_event)
        self.test_infra.dealloc_output()
        trace_input_addr = ttnn.buffer_address(self.test_infra.input_tensor)
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.test_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        assert trace_input_addr == ttnn.buffer_address(self.input_tensor)

        self.device = device

        # More optimized run with caching
        # if use_signpost:
        #    signpost(header="start")

    def get_region_boxes(self, boxes_and_confs):
        print("Getting boxes from boxes and confs ...")
        boxes_list = []
        confs_list = []

        for item in boxes_and_confs:
            boxes_list.append(item[0])
            confs_list.append(item[1])

        # boxes: [batch, num1 + num2 + num3, 1, 4]
        # confs: [batch, num1 + num2 + num3, num_classes]
        boxes = torch.cat(boxes_list, dim=1)
        confs = torch.cat(confs_list, dim=1)

        return [boxes, confs]

    def execute_yolov4_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        ttnn.record_event(1, self.write_event)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        ttnn.record_event(0, self.op_event)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_devices(self.device)
        output = self.test_infra.output_tensor

        output_tensor1 = ttnn.to_torch(output[0])
        output_tensor1 = output_tensor1.reshape(1, 40, 40, 255)
        output_tensor1 = torch.permute(output_tensor1, (0, 3, 1, 2))

        output_tensor2 = ttnn.to_torch(output[1])
        output_tensor2 = output_tensor2.reshape(1, 20, 20, 255)
        output_tensor2 = torch.permute(output_tensor2, (0, 3, 1, 2))

        output_tensor3 = ttnn.to_torch(output[2])
        output_tensor3 = output_tensor3.reshape(1, 10, 10, 255)
        output_tensor3 = torch.permute(output_tensor3, (0, 3, 1, 2))

        n_classes = 80

        yolo1 = YoloLayer(
            anchor_mask=[0, 1, 2],
            num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9,
            stride=8,
        )

        yolo2 = YoloLayer(
            anchor_mask=[3, 4, 5],
            num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9,
            stride=16,
        )

        yolo3 = YoloLayer(
            anchor_mask=[6, 7, 8],
            num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9,
            stride=32,
        )

        y1 = yolo1(output_tensor1)
        y2 = yolo2(output_tensor2)
        y3 = yolo3(output_tensor3)

        output = self.get_region_boxes([y1, y2, y3])

        return output
        # return self.test_infra.output_tensor

        # if use_signpost:
        #    signpost(header="stop")

    def release_yolov4_trace_2cqs_inference(self):
        ttnn.release_trace(self.device, self.tid)

    def run_traced_inference(self, torch_input_tensor):
        n, h, w, c = torch_input_tensor.shape
        torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
        return self.execute_yolov4_trace_2cqs_inference(tt_inputs_host)
