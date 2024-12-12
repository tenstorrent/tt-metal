# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import ttnn
import pytest

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import (
    verify_with_pcc,
    check_pcc_conv,
    is_n300_with_eth_dispatch_cores,
    is_t3k_with_eth_dispatch_cores,
    UNET_FULL_MODEL_PCC,
)

from models.utility_functions import skip_for_grayskull, divup

L1_SMALL_SIZE = 79104
TRACE_REGION_SIZE = 444416


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": L1_SMALL_SIZE, "trace_region_size": TRACE_REGION_SIZE}], indirect=True
)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((1, 2, 128),),
)
def test_unet_trace(
    batch: int,
    groups: int,
    iterations: int,
    device,
    use_program_cache,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups, channel_order="first", pad=False)

    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)
    torch_output_tensor = model(torch_input)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(ttnn_input.volume() // ttnn_input.shape[-1], dram_grid_size.x),
            ttnn_input.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )
    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, dram_memory_config
    )

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)
    logger.info(f"Done compile run")

    logger.info(f"Capturing trace")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)

    input_trace_addr = l1_input_tensor.buffer_address()
    shape = l1_input_tensor.shape
    dtype = l1_input_tensor.dtype
    layout = l1_input_tensor.layout
    output_tensor.deallocate(force=True)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)

    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    l1_input_tensor = ttnn.allocate_tensor_on_device(
        shape, dtype, layout, device, ttnn_model.input_sharded_memory_config
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert input_trace_addr == l1_input_tensor.buffer_address()

    logger.info(f"Running trace for {iterations} iterations...")
    outputs = []
    start = time.time()
    for _ in range(iterations):
        ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=0)
        l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        outputs.append(output_tensor.cpu(blocking=True))
    ttnn.synchronize_device(device)
    end = time.time()
    logger.info(f"Average model performance={iterations * batch / (end-start) : .2f} fps")

    logger.info(f"Running sanity check against reference model output")

    B, C, H, W = torch_output_tensor.shape
    verify_with_pcc(torch_output_tensor, ttnn.to_torch(outputs[-1]).reshape(B, C, H, W), pcc=UNET_FULL_MODEL_PCC)


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": L1_SMALL_SIZE, "trace_region_size": TRACE_REGION_SIZE}], indirect=True
)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((1, 2, 128),),
)
def test_unet_trace_2cq(
    batch: int,
    groups: int,
    iterations: int,
    device,
    use_program_cache,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups, pad=False)

    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)
    torch_output_tensor = model(torch_input)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)

    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(ttnn_input.volume() // ttnn_input.shape[-1], dram_grid_size.x),
            ttnn_input.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, dram_memory_config
    )
    ttnn.record_event(0, op_event)

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)

    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)

    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)
    logger.info(f"Done compile run")

    logger.info(f"Capturing trace")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)

    input_trace_addr = l1_input_tensor.buffer_address()
    shape = l1_input_tensor.shape
    dtype = l1_input_tensor.dtype
    layout = l1_input_tensor.layout
    output_tensor.deallocate(force=True)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)

    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    l1_input_tensor = ttnn.allocate_tensor_on_device(
        shape, dtype, layout, device, ttnn_model.input_sharded_memory_config
    )
    # assert input_trace_addr == l1_input_tensor.buffer_address()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    outputs = []
    start = time.time()
    for _ in range(iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)

        l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
        ttnn.record_event(0, op_event)

        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(output_tensor.cpu(blocking=False))
    ttnn.synchronize_device(device)
    end = time.time()
    logger.info(f"Average model time={1000.0 * (end-start) / iterations : .2f} ms")
    logger.info(f"Average model performance={iterations * groups * batch / (end-start) : .2f} fps")

    ttnn.DumpDeviceProfiler(device)

    logger.info(f"Running sanity check against reference model output")
    B, C, H, W = torch_output_tensor.shape
    verify_with_pcc(torch_output_tensor, ttnn.to_torch(outputs[-1]).reshape(B, C, H, W), pcc=UNET_FULL_MODEL_PCC)

    ttnn.release_trace(device, tid)


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("enable_async_mode", (True,), indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 68864, "trace_region_size": 444416, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((1, 2, 128),),
)
def test_unet_trace_2cq_multi_device(
    batch: int, groups: int, iterations: int, mesh_device, use_program_cache, reset_seeds, enable_async_mode
):
    if not is_n300_with_eth_dispatch_cores(mesh_device) and not is_t3k_with_eth_dispatch_cores(mesh_device):
        pytest.skip("Test is only valid for N300 or T3000")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device=mesh_device, mesh_mapper=weights_mesh_mapper)

    num_devices = len(mesh_device.get_device_ids())
    logger.info(f"Using {num_devices} devices for this test")

    total_batch = num_devices * batch
    torch_input, ttnn_input = create_unet_input_tensors(total_batch, groups, mesh_mapper=inputs_mesh_mapper)
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input.shape)} on devices={mesh_device.get_device_ids()}"
    )

    torch_output_tensor = model(torch_input)

    op_event = ttnn.create_event(mesh_device)
    write_event = ttnn.create_event(mesh_device)

    dram_grid_size = mesh_device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(ttnn_input.volume() // ttnn_input.shape[-1], dram_grid_size.x),
            ttnn_input.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device, dram_memory_config
    )
    ttnn.record_event(0, op_event)

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)

    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)

    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)
    logger.info(f"Done compile run")

    logger.info(f"Capturing trace")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)

    input_trace_addr = buffer_address(l1_input_tensor)
    shape = l1_input_tensor.shape
    dtype = l1_input_tensor.dtype
    layout = l1_input_tensor.layout
    output_tensor.deallocate(force=True)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)

    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    l1_input_tensor = ttnn.allocate_tensor_on_device(
        shape, dtype, layout, mesh_device, ttnn_model.input_sharded_memory_config
    )
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    assert input_trace_addr == buffer_address(l1_input_tensor)

    outputs = []
    start = time.time()
    for _ in range(iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)

        l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
        ttnn.record_event(0, op_event)

        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        outputs.append(output_tensor.cpu(blocking=False))

    ttnn.synchronize_devices(mesh_device)

    end = time.time()
    logger.info(f"Average model performance={iterations * groups * total_batch / (end-start) : .2f} fps")

    logger.info(f"Running sanity check against reference model output")
    B, C, H, W = torch_output_tensor.shape
    verify_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(outputs[-1], mesh_composer=output_mesh_composer).reshape(B, C, H, W),
        pcc=UNET_FULL_MODEL_PCC,
    )

    ttnn.release_trace(mesh_device, tid)


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 68864, "trace_region_size": 444416, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((1, 2, 128),),
)
def test_unet_trace_2cq_same_io(
    batch: int,
    groups: int,
    iterations: int,
    device,
    use_program_cache,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)

    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)
    torch_output_tensor = model(torch_input)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    model_event = ttnn.create_event(device)
    read_event = ttnn.create_event(device)

    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(ttnn_input.volume() // ttnn_input.shape[-1], dram_grid_size.x),
            ttnn_input.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    input_dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, input_dram_memory_config
    )
    ttnn.record_event(0, op_event)
    ttnn.record_event(1, read_event)

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)

    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)

    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            output_tensor.volume() // output_tensor.shape[-1],
            divup(output_tensor.shape[-1], dram_grid_size.x),
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    dram_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config)
    logger.info(f"Done compile run")

    logger.info(f"Capturing trace")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)

    input_trace_addr = l1_input_tensor.buffer_address()
    shape = l1_input_tensor.shape
    dtype = l1_input_tensor.dtype
    layout = l1_input_tensor.layout
    output_tensor.deallocate(force=True)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)

    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    l1_input_tensor = ttnn.allocate_tensor_on_device(
        shape, dtype, layout, device, ttnn_model.input_sharded_memory_config
    )
    # assert input_trace_addr == l1_input_tensor.buffer_address()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)
    ttnn.synchronize_device(device)

    outputs = []
    start = time.time()
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
    ttnn.record_event(1, write_event)
    for _ in range(iterations - 1):
        ttnn.wait_for_event(0, write_event)
        l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)
        dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)
        ttnn.record_event(0, model_event)
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(1, model_event)
        outputs.append(dram_output_tensor.cpu(blocking=False, cq_id=1))
        ttnn.record_event(1, read_event)
    ttnn.wait_for_event(0, write_event)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
    ttnn.record_event(0, op_event)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.wait_for_event(0, read_event)
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)
    ttnn.record_event(0, model_event)
    ttnn.wait_for_event(1, model_event)
    outputs.append(dram_output_tensor.cpu(blocking=False, cq_id=1))
    ttnn.synchronize_device(device)
    end = time.time()
    logger.info(f"Average model time={1000.0 * (end-start) / iterations : .2f} ms")
    logger.info(f"Average model performance={iterations * groups * batch / (end-start) : .2f} fps")

    logger.info(f"Running sanity check against reference model output")
    B, C, H, W = torch_output_tensor.shape
    verify_with_pcc(torch_output_tensor, ttnn.to_torch(outputs[-1]).reshape(B, C, H, W), pcc=UNET_FULL_MODEL_PCC)
    ttnn.release_trace(device, tid)


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("enable_async_mode", (True,), indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 68864, "trace_region_size": 444416, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((1, 2, 128),),
)
def test_unet_trace_2cq_same_io_multi_device(
    batch: int, groups: int, iterations: int, mesh_device, use_program_cache, reset_seeds, enable_async_mode
):
    if not is_n300_with_eth_dispatch_cores(mesh_device) and not is_t3k_with_eth_dispatch_cores(mesh_device):
        pytest.skip("Test is only valid for N300 or T3000")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device=mesh_device, mesh_mapper=weights_mesh_mapper)

    num_devices = len(mesh_device.get_device_ids())
    logger.info(f"Using {num_devices} devices for this test")

    total_batch = num_devices * batch
    torch_input, ttnn_input = create_unet_input_tensors(total_batch, groups, mesh_mapper=inputs_mesh_mapper)
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input.shape)} on devices={mesh_device.get_device_ids()}"
    )

    torch_output_tensor = model(torch_input)

    op_event = ttnn.create_event(mesh_device)
    write_event = ttnn.create_event(mesh_device)
    model_event = ttnn.create_event(mesh_device)
    read_event = ttnn.create_event(mesh_device)

    dram_grid_size = mesh_device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(ttnn_input.volume() // ttnn_input.shape[-1], dram_grid_size.x),
            ttnn_input.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    input_dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    input_tensor = ttnn.allocate_tensor_on_device(
        ttnn_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device, input_dram_memory_config
    )
    ttnn.record_event(0, op_event)
    ttnn.record_event(1, read_event)

    logger.info(f"Compiling model with warmup run")
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)

    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)

    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            output_tensor.volume() // output_tensor.shape[-1],
            divup(output_tensor.shape[-1], dram_grid_size.x),
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    dram_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config)
    logger.info(f"Done compile run")

    logger.info(f"Capturing trace")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config)
    ttnn.record_event(0, op_event)

    input_trace_addr = buffer_address(l1_input_tensor)
    shape = l1_input_tensor.shape
    dtype = l1_input_tensor.dtype
    layout = l1_input_tensor.layout
    output_tensor.deallocate(force=True)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output_tensor = ttnn_model(l1_input_tensor, move_input_tensor_to_device=False)

    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    l1_input_tensor = ttnn.allocate_tensor_on_device(
        shape, dtype, layout, mesh_device, ttnn_model.input_sharded_memory_config
    )
    # assert input_trace_addr == l1_input_tensor.buffer_address()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)
    ttnn.synchronize_devices(mesh_device)

    outputs = []
    start = time.time()
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
    ttnn.record_event(1, write_event)
    for _ in range(iterations - 1):
        ttnn.wait_for_event(0, write_event)
        l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)
        dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)
        ttnn.record_event(0, model_event)
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(ttnn_input, input_tensor, cq_id=1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(1, model_event)
        outputs.append(dram_output_tensor.cpu(blocking=False, cq_id=1))
        ttnn.record_event(1, read_event)
    ttnn.wait_for_event(0, write_event)
    l1_input_tensor = ttnn.reshard(input_tensor, ttnn_model.input_sharded_memory_config, l1_input_tensor)
    ttnn.record_event(0, op_event)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.wait_for_event(0, read_event)
    dram_output_tensor = ttnn.reshard(output_tensor, dram_memory_config, dram_output_tensor)
    ttnn.record_event(0, model_event)
    ttnn.wait_for_event(1, model_event)
    outputs.append(dram_output_tensor.cpu(blocking=False, cq_id=1))
    ttnn.synchronize_devices(mesh_device)
    end = time.time()
    logger.info(f"Average model time={1000.0 * (end-start) / iterations : .2f} ms")
    logger.info(f"Average model performance={iterations * groups * total_batch / (end-start) : .2f} fps")

    logger.info(f"Running sanity check against reference model output")
    B, C, H, W = torch_output_tensor.shape
    verify_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(outputs[-1], mesh_composer=output_mesh_composer).reshape(B, C, H, W),
        pcc=UNET_FULL_MODEL_PCC,
    )
    ttnn.release_trace(mesh_device, tid)
