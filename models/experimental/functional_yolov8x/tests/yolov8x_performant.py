# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import pytest
from models.experimental.functional_yolov8x.tests.yolov8x_test_infra import create_test_infra

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


def run_yolov8x_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
    )

    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    # # First run configures convs JIT
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
    act_dtype,
    weight_dtype,
):
    test_infra = create_test_infra(
        device=device,
        batch_size=device_batch_size,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
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
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(tt_image_res)

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
    act_dtype,
    weight_dtype,
    model_location_generator=None,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        # model_location_generator=model_location_generator,
    )
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
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
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
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
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.dealloc_output()
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    input_tensor = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(input_tensor)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    for iter in range(0, 2):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        input_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_devices(device)

    if use_signpost:
        signpost(header="stop")

    ttnn.release_trace(device, tid)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 800768}], indirect=True)
@pytest.mark.parametrize("device_batch_size, act_dtype, weight_dtype", [(1, ttnn.bfloat16, ttnn.bfloat8_b)])
def test_run_yolov8x_trace_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
):
    run_yolov8x_trace_inference(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 3686400, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_yolov8x_trace_2cqs_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    enable_async_mode,
):
    run_yolov8x_trace_2cqs_inference(
        device=device,
        device_batch_size=device_batch_size,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("device_batch_size, act_dtype, weight_dtype", [(1, ttnn.bfloat16, ttnn.bfloat8_b)])
def test_run_yolov8x_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
):
    run_yolov8x_inference(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
    )
