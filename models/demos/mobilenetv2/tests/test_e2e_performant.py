# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tests.mobilenetv2_common import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE
from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
    get_mesh_mappers,
)
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_mobilenetv2_e2e(
    device,
    batch_size,
    model_location_generator=None,
):
    num_devices = device.get_num_devices()
    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
    if batch_size % num_devices != 0:
        raise ValueError(f"Batch size {batch_size} is not divisible by number of devices {num_devices}")
    batch_size_per_device = batch_size // num_devices
    torch_input_tensor, host_input_tensor = create_mobilenetv2_input_tensors(
        batch=batch_size, input_height=224, input_width=224, pad_channels=16, mesh_mapper=inputs_mesh_mapper
    )

    torch_model = Mobilenetv2()
    torch_model.eval()
    torch_output_tensor = torch_model(torch_input_tensor)

    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size_per_device)

    dram_cores = 10
    assert host_input_tensor.shape[-2] % dram_cores == 0, "Expecting even sharding on DRAM input tensor"
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}),
        [host_input_tensor.shape[-2] // dram_cores, host_input_tensor.shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_dram_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
    assert host_input_tensor.shape[-2] % input_l1_core_grid.num_cores == 0, "Expecting even sharding on L1 input tensor"
    input_l1_mem_config = ttnn.create_sharded_memory_config(
        shape=(host_input_tensor.shape[2] // input_l1_core_grid.num_cores, host_input_tensor.shape[-1]),
        core_grid=input_l1_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_dram_tensor = ttnn.allocate_tensor_on_device(
        host_input_tensor.shape, host_input_tensor.dtype, host_input_tensor.layout, device, input_dram_mem_config
    )

    op_event = ttnn.record_event(device, 0)

    logger.info(f"Compiling model")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(host_input_tensor, input_dram_tensor, cq_id=1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    input_l1_tensor = ttnn.reshard(input_dram_tensor, input_l1_mem_config)
    op_event = ttnn.record_event(device, 0)
    output_tensor = ttnn_model(input_l1_tensor)

    logger.info(f"Capturing trace of model")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(host_input_tensor, input_dram_tensor, cq_id=1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    input_l1_tensor = ttnn.reshard(input_dram_tensor, input_l1_mem_config)
    op_event = ttnn.record_event(device, 0)
    input_trace_addr = input_l1_tensor.buffer_address()
    output_tensor.deallocate(force=True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = ttnn_model(input_l1_tensor)
    input_l1_tensor = ttnn.allocate_tensor_on_device(input_l1_tensor.spec, device)
    assert input_trace_addr == input_l1_tensor.buffer_address()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    outputs = []
    iterations = 32
    logger.info(f"Trace captured - running model benchmarks for {iterations} iterations...")
    start = time.time()
    for _ in range(iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(host_input_tensor, input_dram_tensor, cq_id=1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        input_l1_tensor = ttnn.reshard(input_dram_tensor, input_l1_mem_config, input_l1_tensor)
        op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(output_tensor.cpu(blocking=False))
    ttnn.synchronize_device(device)
    end = time.time()

    inference_time = (end - start) / iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={iterations * batch_size / (end-start) : .2f} fps")
    assert_with_pcc(torch_output_tensor, ttnn.to_torch(outputs[-1], mesh_composer=output_mesh_composer), 0.99)


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
def test_mobilenetv2_e2e(batch_size, device):
    run_mobilenetv2_e2e(device, batch_size)


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((2 * MOBILENETV2_BATCH_SIZE),),
)
def test_mobilenetv2_e2e_dp(batch_size, mesh_device):
    run_mobilenetv2_e2e(mesh_device, batch_size)
