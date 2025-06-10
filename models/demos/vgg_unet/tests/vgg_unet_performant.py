# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.vgg_unet.tests.vgg_unet_test_infra import create_test_infra

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def run_vgg_unet_inference(
    device,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        model_location_generator=model_location_generator,
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


def run_vgg_unet_trace_inference(
    device,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        model_location_generator=model_location_generator,
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
