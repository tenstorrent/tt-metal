# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov8x.runner.performant_runner_infra import YOLOv8xPerformanceRunnerInfra

try:
    from tracy import signpost

    use_signpost = True

except ModuleNotFoundError:
    use_signpost = False


def run_yolov8x_inference(
    device,
    device_batch_size,
):
    test_infra = YOLOv8xPerformanceRunnerInfra(
        device=device,
        batch_size=device_batch_size,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()
    test_infra.dealloc_output()


def run_yolov8x_trace_inference(
    device,
    device_batch_size,
):
    test_infra = YOLOv8xPerformanceRunnerInfra(
        device=device,
        batch_size=device_batch_size,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    spec = test_infra.input_tensor.spec
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()

    # Capture
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)

    test_infra.dealloc_output()
    trace_input_addr = test_infra.input_tensor.buffer_address()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == tt_image_res.buffer_address()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()

    ttnn.release_trace(device, tid)
    test_infra.dealloc_output()


def run_yolov8x_trace_2cqs_inference(
    device,
    device_batch_size,
):
    test_infra = YOLOv8xPerformanceRunnerInfra(
        device,
        device_batch_size,
    )
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)

    # Initialize the op event so we can write
    op_event = ttnn.record_event(device, 0)

    # First run configures convs JIT
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    spec = test_infra.input_tensor.spec
    op_event = ttnn.record_event(device, 0)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # Optimized run
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    test_infra.run()
    test_infra.validate()

    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    test_infra.dealloc_output()
    trace_input_addr = test_infra.input_tensor.buffer_address()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    input_tensor = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == input_tensor.buffer_address()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    for iter in range(0, 2):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        input_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_tensor)
        op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    if use_signpost:
        signpost(header="stop")

    ttnn.release_trace(device, tid)
