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


# TODO: Move these into Resnet model preprocessing/member functions
def setup_l1_sharded_input(device, tt_inputs, tt_resnet50, mesh_mapper, mesh_composer):
    num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()

    padded_input_shape, input_mem_config, _ = ttnn.get_conv_padded_input_shape_and_mem_config(
        device=device,
        input_tensor=tt_inputs,
        conv_config=tt_resnet50.conv1_config,
        batch_size=tt_resnet50.batch_size,
        height=tt_resnet50.conv1_output_height,
        width=tt_resnet50.conv1_output_width,
        in_channels=tt_resnet50.conv1_input_channels,
        out_channels=tt_resnet50.conv1_output_channels,
    )

    inputs_padded = ttnn.to_torch(tt_inputs, device=device, mesh_composer=mesh_composer)
    inputs_padded = inputs_padded.reshape(num_devices, 1, -1, inputs_padded.shape[-1])
    inputs_padded = torch.nn.functional.pad(
        inputs_padded,
        (0, padded_input_shape[-1] - inputs_padded.shape[-1], 0, padded_input_shape[-2] - inputs_padded.shape[-2]),
    )
    tt_inputs_host = ttnn.from_torch(
        inputs_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper
    )
    return tt_inputs_host, input_mem_config


def setup_dram_sharded_input(device, tt_inputs, tt_resnet50, mesh_mapper, mesh_composer):
    tt_inputs_host, input_mem_config = setup_l1_sharded_input(
        device, tt_inputs, tt_resnet50, mesh_mapper, mesh_composer
    )
    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x),
            tt_inputs_host.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    sharded_mem_config_DRAM = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", [True, False], indirect=True)
def test_run_resnet50_inference(
    device_mesh,
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

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
    output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)

    test_infra = create_test_infra(
        device_mesh,
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

    test_infra.preprocess_torch_input()
    tt_inputs_host, input_mem_config = setup_l1_sharded_input(
        device_mesh,
        test_infra.input_tensor,
        test_infra.ttnn_resnet50_model,
        inputs_mesh_mapper,
        output_mesh_composer,
    )

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device_mesh, input_mem_config)
    test_infra.run()
    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device_mesh, input_mem_config)
    test_infra.run()
    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    test_infra.input_tensor = tt_inputs_host.to(device_mesh, input_mem_config)
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
    device_mesh,
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

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
    output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)

    test_infra = create_test_infra(
        device_mesh,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        True,
        final_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        model_location_generator=model_location_generator,
    )
    test_infra.preprocess_torch_input()
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = setup_dram_sharded_input(
        device_mesh,
        test_infra.input_tensor,
        test_infra.ttnn_resnet50_model,
        inputs_mesh_mapper,
        output_mesh_composer,
    )
    tt_image_res = tt_inputs_host.to(device_mesh, sharded_mem_config_DRAM)

    # First run configures convs JIT
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    test_infra.run()

    # Optimized run
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    test_infra.run()

    # Capture
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    tid = ttnn.begin_trace_capture(device_mesh, cq_id=0)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    test_infra.run()
    ttnn.end_trace_capture(device_mesh, tid, cq_id=0)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    ttnn.execute_trace(device_mesh, tid, cq_id=0, blocking=True)
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
    device_mesh,
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

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
    output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)

    test_infra = create_test_infra(
        device_mesh,
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
    test_infra.preprocess_torch_input()
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = setup_dram_sharded_input(
        device_mesh,
        test_infra.input_tensor,
        test_infra.ttnn_resnet50_model,
        inputs_mesh_mapper,
        output_mesh_composer,
    )
    tt_image_res = tt_inputs_host.to(device_mesh, sharded_mem_config_DRAM)
    op_event = ttnn.create_event(device_mesh)
    write_event = ttnn.create_event(device_mesh)
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

    ttnn.synchronize_devices(device_mesh)

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
    device_mesh,
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

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
    output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)

    test_infra = create_test_infra(
        device_mesh,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        True,
        final_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        model_location_generator=model_location_generator,
    )
    test_infra.preprocess_torch_input()
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = setup_dram_sharded_input(
        device_mesh,
        test_infra.input_tensor,
        test_infra.ttnn_resnet50_model,
        inputs_mesh_mapper,
        output_mesh_composer,
    )
    tt_image_res = tt_inputs_host.to(device_mesh, sharded_mem_config_DRAM)
    op_event = ttnn.create_event(device_mesh)
    write_event = ttnn.create_event(device_mesh)
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
    first_out_addr = ttnn.buffer_address(test_infra.input_tensor)
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
    tid = ttnn.begin_trace_capture(device_mesh, cq_id=0)
    test_infra.run()
    test_infra.input_tensor = ttnn.allocate_tensor_on_device(
        test_infra.input_tensor.shape,
        test_infra.input_tensor.dtype,
        test_infra.input_tensor.layout,
        device_mesh,
        input_mem_config,
    )
    ttnn.end_trace_capture(device_mesh, tid, cq_id=0)
    assert first_out_addr == ttnn.buffer_address(test_infra.input_tensor)
    test_infra.validate()

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
        test_infra.input_tensor = ttnn.reshard(tt_image_res, input_mem_config, test_infra.input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device_mesh, tid, cq_id=0, blocking=False)
        outputs.append(ttnn.from_device(test_infra.output_tensor, blocking=False))
    ttnn.synchronize_devices(device_mesh)

    if use_signpost:
        signpost(header="stop")
    for output in outputs:
        test_infra.validate(output)
