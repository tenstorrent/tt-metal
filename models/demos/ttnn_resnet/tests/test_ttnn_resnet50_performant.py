# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import (
    is_wormhole_b0,
    divup,
    skip_for_grayskull,
)
from models.demos.ttnn_resnet.tests.ttnn_resnet_test_infra import create_test_infra

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


# TODO: Move these into Resnet model preprocessing/member functions
def setup_l1_sharded_input(device, tt_inputs, tt_resnet50):
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
    inputs_padded = tt_inputs.to_torch()
    inputs_padded = inputs_padded.reshape(1, 1, -1, inputs_padded.shape[-1])
    inputs_padded = torch.nn.functional.pad(
        inputs_padded,
        (0, padded_input_shape[-1] - inputs_padded.shape[-1], 0, padded_input_shape[-2] - inputs_padded.shape[-2]),
    )
    tt_inputs_host = ttnn.from_torch(inputs_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return tt_inputs_host, input_mem_config


def setup_dram_sharded_input(device, tt_inputs, tt_resnet50):
    tt_inputs_host, input_mem_config = setup_l1_sharded_input(device, tt_inputs, tt_resnet50)
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


@skip_for_grayskull(reason_str="Untested for Grayskull")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_inference(
    device, use_program_cache, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
):
    if batch_size == 8:
        pytest.skip("Skipping batch size 8 due to memory config issue")
    if is_wormhole_b0() and batch_size == 20:
        pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")
    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()


@skip_for_grayskull(reason_str="Untested for Grayskull")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 800768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_run_resnet50_trace_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async,
    model_location_generator,
):
    if batch_size == 8:
        pytest.skip("Skipping batch size 8 due to memory config issue")
    if is_wormhole_b0() and batch_size == 20:
        pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")

    device.enable_async(enable_async)

    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)

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
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    test_infra.run()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()

    device.enable_async(False)


@skip_for_grayskull(reason_str="Untested for Grayskull")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_hw_cqs": 2}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_resnet50_2cqs_inference(
    device, use_program_cache, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
):
    if batch_size == 8:
        pytest.skip("Skipping batch size 8 due to memory config issue")
    if is_wormhole_b0() and batch_size == 20:
        pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")
    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
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
    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="stop")
    for output in outputs:
        test_infra.validate(output)


@skip_for_grayskull(reason_str="Untested for Grayskull")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 800768, "num_hw_cqs": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_run_resnet50_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async,
    model_location_generator,
):
    if batch_size == 8:
        pytest.skip("Skipping batch size 8 due to memory config issue")
    if is_wormhole_b0() and batch_size == 20:
        pytest.skip("Skipping batch size 20 for Wormhole B0 due to fitting issue")

    device.enable_async(enable_async)

    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
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
    shape = test_infra.input_tensor.shape
    dtype = test_infra.input_tensor.dtype
    layout = test_infra.input_tensor.layout
    ttnn.record_event(0, op_event)
    test_infra.run()
    test_infra.validate()

    # Optimized run
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    first_out_addr = test_infra.input_tensor.buffer_address()
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
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    input_tensor = ttnn.allocate_tensor_on_device(
        shape,
        dtype,
        layout,
        device,
        input_mem_config,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert first_out_addr == input_tensor.buffer_address()
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
        # TODO: Add in place support to ttnn to_memory_config
        input_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(ttnn.from_device(test_infra.output_tensor, blocking=False))

    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="stop")
    for output in outputs:
        test_infra.validate(output)

    device.enable_async(False)
