# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.demos.mobilenetv2.common import load_torch_model
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tests.perf.mobilenetv2_common import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE
from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
    get_mesh_mappers,
)
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_mobilenetv2_e2e(
    device,
    batch_size_per_device,
    model_location_generator=None,
):
    num_devices = device.get_num_devices()
    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
    batch_size = batch_size_per_device * num_devices
    torch_input_tensor, host_input_tensor = create_mobilenetv2_input_tensors(
        batch=batch_size, input_height=224, input_width=224, pad_channels=16, mesh_mapper=inputs_mesh_mapper
    )
    torch_model = load_torch_model(Mobilenetv2(), model_location_generator=model_location_generator)
    torch_output_tensor = torch_model(torch_input_tensor)

    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size_per_device)

    input_dram_mem_config = get_memory_config_for_persistent_dram_tensor(
        host_input_tensor.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )
    logger.info(
        f"Auto-selected persistent DRAM tensor memory config: shape={host_input_tensor.shape}, shard_shape={input_dram_mem_config.shard_spec.shape}, grid={input_dram_mem_config.shard_spec.grid}"
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

    config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipe = create_pipeline_from_config(
        config,
        ttnn_model,
        device,
        dram_input_memory_config=input_dram_mem_config,
        l1_input_memory_config=input_l1_mem_config,
    )

    iterations = 32
    host_inputs = [host_input_tensor] * iterations

    pipe.compile(host_input_tensor)
    pipe.preallocate_output_tensors_on_host(len(host_inputs))

    start = time.time()
    outputs = pipe.enqueue(host_inputs).pop_all()
    end = time.time()

    pipe.cleanup()

    inference_time = (end - start) / iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={iterations * batch_size / (end-start) : .2f} fps")
    assert_with_pcc(torch_output_tensor, ttnn.to_torch(outputs[-1], mesh_composer=output_mesh_composer), 0.94)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
def test_mobilenetv2_e2e(batch_size, device, model_location_generator):
    run_mobilenetv2_e2e(device, batch_size, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
def test_mobilenetv2_e2e_dp(device_batch_size, mesh_device, model_location_generator):
    run_mobilenetv2_e2e(mesh_device, device_batch_size, model_location_generator)
