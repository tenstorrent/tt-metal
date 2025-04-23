# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.ufld_v2.tests.ufld_v2_test_infra import create_test_infra


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


ttnn.buffer_address = buffer_address


def run_ufld_v2_inference(
    device,
    device_batch_size,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
    )

    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)

    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    if use_signpost:
        signpost(header="start")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()
    test_infra.dealloc_output()


def run_ufld_v2_trace_inference(
    device,
    device_batch_size,
):
    test_infra = create_test_infra(
        device=device,
        batch_size=device_batch_size,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    spec = test_infra.input_tensor.spec
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    test_infra.validate()

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)

    test_infra.dealloc_output()
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(tt_image_res)

    if use_signpost:
        signpost(header="start")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()

    ttnn.release_trace(device, tid)
    test_infra.dealloc_output()


def ufld_v2_trace_2cqs_inference(
    device,
    device_batch_size,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
    )
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)

    op_event = ttnn.record_event(device, 0)

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

    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    test_infra.run()
    test_infra.validate()

    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    test_infra.dealloc_output()
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    input_tensor = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(input_tensor)

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
