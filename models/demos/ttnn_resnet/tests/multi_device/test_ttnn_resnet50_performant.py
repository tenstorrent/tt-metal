# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import (
    is_wormhole_b0,
    divup,
    run_for_wormhole_b0,
)
from models.demos.ttnn_resnet.tests.ttnn_resnet_test_infra import create_test_infra

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


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_inference(
    mesh_device,
    use_program_cache,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    if device_batch_size == 8:
        pytest.skip("Skipping batch size 8 due to memory config issue")
    if is_wormhole_b0() and device_batch_size == 20:
        pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    test_infra = create_test_infra(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        model_location_generator=model_location_generator,
    )

    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(
        mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)
    test_infra.run()
    test_infra.validate()

    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)
    test_infra.run()
    test_infra.validate()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)
    test_infra.run()
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 800768}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_trace_inference(
    mesh_device,
    use_program_cache,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    if device_batch_size == 8:
        pytest.skip("Skipping batch size 8 due to memory config issue")
    if is_wormhole_b0() and device_batch_size == 20:
        pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    test_infra = create_test_infra(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(
        mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)
    shape = test_infra.input_tensor.shape
    dtype = test_infra.input_tensor.dtype
    layout = test_infra.input_tensor.layout
    test_infra.run()
    test_infra.validate()
    test_infra.output_tensor.deallocate(force=True)

    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)
    test_infra.run()
    test_infra.validate()

    # Capture
    test_infra.input_tensor = tt_inputs_host.to(mesh_device, input_mem_config)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    test_infra.run()
    tt_image_res = ttnn.allocate_tensor_on_device(
        shape,
        dtype,
        layout,
        mesh_device,
        input_mem_config,
    )
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(tt_image_res)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_2cqs_inference(
    mesh_device,
    use_program_cache,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    if device_batch_size == 8:
        pytest.skip("Skipping batch size 8 due to memory config issue")
    if is_wormhole_b0() and device_batch_size == 20:
        pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    test_infra = create_test_infra(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(
        mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_image_res = tt_inputs_host.to(mesh_device, sharded_mem_config_DRAM)
    op_event = ttnn.create_event(mesh_device)
    write_event = ttnn.create_event(mesh_device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    # First run configures convs JIT
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # Optimized run
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    outputs = []
    for iter in range(0, 2):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
        ttnn.record_event(0, op_event)
        outputs.append(ttnn.from_device(test_infra.run(), blocking=False))

    ttnn.synchronize_devices(mesh_device)

    if use_signpost:
        signpost(header="stop")
    for output in outputs:
        test_infra.validate(output)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 800768, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_trace_2cqs_inference(
    mesh_device,
    use_program_cache,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    if device_batch_size == 8:
        pytest.skip("Skipping batch size 8 due to memory config issue")
    if is_wormhole_b0() and device_batch_size == 20:
        pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    test_infra = create_test_infra(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(
        mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_image_res = tt_inputs_host.to(mesh_device, sharded_mem_config_DRAM)
    op_event = ttnn.create_event(mesh_device)
    write_event = ttnn.create_event(mesh_device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    # First run configures convs JIT
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    shape = test_infra.input_tensor.shape
    dtype = test_infra.input_tensor.dtype
    layout = test_infra.input_tensor.layout
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()
    test_infra.output_tensor.deallocate(force=True)

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
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    test_infra.run()
    input_tensor = ttnn.allocate_tensor_on_device(
        shape,
        dtype,
        layout,
        mesh_device,
        input_mem_config,
    )
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(input_tensor)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    outputs = []
    for iter in range(0, 1):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        input_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        outputs.append(ttnn.from_device(test_infra.output_tensor, blocking=False))
    ttnn.synchronize_devices(mesh_device)

    if use_signpost:
        signpost(header="stop")
    for output in outputs:
        test_infra.validate(output)
