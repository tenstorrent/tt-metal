# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import ttnn
import pytest

from ttnn.device import is_wormhole_b0

from loguru import logger

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import (
    verify_with_pcc,
    UNET_FULL_MODEL_PCC,
    UNET_FULL_MODEL_PCC_BH,
    UNET_TRACE_REGION_SIZE,
    UNET_L1_SMALL_REGION_SIZE,
)
from models.tt_cnn.tests.test_executor import create_pipeline_from_config, PipelineConfig
from models.experimental.functional_unet.tests.test_unet_trace import (
    determine_num_cores_for_even_sharding,
    get_dram_sharded_memory_config_for_tensor,
)

from models.utility_functions import skip_for_grayskull


def get_l1_sharded_memory_config_for_tensor(tensor, grid_size):
    """Helper function to create a sharded memory config for L1."""
    total_number_of_l1_cores = grid_size.x * grid_size.y
    l1_cores_for_even_sharding = determine_num_cores_for_even_sharding(tensor.shape[-1], total_number_of_l1_cores)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(l1_cores_for_even_sharding - 1, 0))}),
        [tensor.shape[-2], tensor.shape[-1] // l1_cores_for_even_sharding],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": UNET_L1_SMALL_REGION_SIZE,
            "trace_region_size": UNET_TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((1, 4, 128),),
)
def test_unet_pipeline_2cq(
    batch: int,
    groups: int,
    iterations: int,
    device,
    reset_seeds,
):
    # Prepare inputs and reference output
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups, channel_order="first", pad=False, fold=True)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)
    torch_output_tensor = model(torch_input)

    # Prepare ttnn model
    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    def model_wrapper(l1_input_tensor, ttnn_model=ttnn_model):
        return ttnn_model(l1_input_tensor, move_input_tensor_to_device=False, deallocate_input_activation=False)

    # Prepare memory configs
    dram_grid_size = device.dram_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()

    dram_memory_config = get_dram_sharded_memory_config_for_tensor(ttnn_input, dram_grid_size)
    l1_memory_config = ttnn_model.input_sharded_memory_config

    # Hardcode output memory config as requested
    output_dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
        [4, 14080],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, output_dram_shard_spec
    )

    # Create pipeline
    config = PipelineConfig(use_trace=True, num_command_queues=2, separate_io_queue=True)
    pipe = create_pipeline_from_config(
        config,
        model_wrapper,
        device,
        dram_input_memory_config=dram_memory_config,
        l1_input_memory_config=l1_memory_config,
        dram_output_memory_config=dram_output_memory_config,
        output_shape=[1, 1, 4, 168960],
        output_dtype=ttnn.bfloat16,
    )

    # Prepare host inputs
    host_inputs = []
    for _ in range(iterations):
        _, input_tensor = create_unet_input_tensors(batch, groups, channel_order="first", pad=False, fold=True)
        host_inputs.append(input_tensor)

    # Run pipeline
    logger.info("Compiling model and recording trace...")
    pipe.compile(host_inputs[0])

    pipe.preallocate_output_tensors_on_host(
        iterations,
        output_shape=[1, 1, 4, 168960],
        output_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dram_output_memory_config=dram_output_memory_config,
    )

    logger.info(f"Running model for {iterations} iterations")
    start = time.time()
    outputs = pipe.enqueue_fast(host_inputs).pop_all_fast()
    end = time.time()
    inference_time = (end - start) / iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={iterations * groups * batch / (end-start) : .2f} fps")

    pipe.cleanup()

    # Verify outputs
    assert len(outputs) == len(host_inputs)
    B, C, H, W = torch_output_tensor.shape
    for i in range(len(outputs)):
        verify_with_pcc(
            torch_output_tensor,
            ttnn.to_torch(outputs[i]).reshape(B, C, H, W),
            pcc=UNET_FULL_MODEL_PCC if is_wormhole_b0(device) else UNET_FULL_MODEL_PCC_BH,
        )
    logger.info("PCC check passed.")


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": UNET_L1_SMALL_REGION_SIZE,
            "trace_region_size": UNET_TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch, groups, iterations",
    ((1, 4, 128),),
)
def test_unet_pipeline_2cq_optimized(
    batch: int,
    groups: int,
    iterations: int,
    device,
    reset_seeds,
):
    """Optimized pipeline test using fast path execution"""
    # Prepare inputs and reference output
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups, channel_order="first", pad=False, fold=True)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)
    torch_output_tensor = model(torch_input)

    # Prepare ttnn model
    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    def model_wrapper(l1_input_tensor, ttnn_model=ttnn_model):
        return ttnn_model(l1_input_tensor, move_input_tensor_to_device=False, deallocate_input_activation=False)

    # Prepare memory configs
    dram_grid_size = device.dram_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()

    dram_memory_config = get_dram_sharded_memory_config_for_tensor(ttnn_input, dram_grid_size)
    l1_memory_config = ttnn_model.input_sharded_memory_config

    # Hardcode output memory config as requested
    output_dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
        [4, 14080],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, output_dram_shard_spec
    )

    # Create pipeline
    config = PipelineConfig(use_trace=True, num_command_queues=2, separate_io_queue=True)
    pipe = create_pipeline_from_config(
        config,
        model_wrapper,
        device,
        dram_input_memory_config=dram_memory_config,
        l1_input_memory_config=l1_memory_config,
        dram_output_memory_config=dram_output_memory_config,
        output_shape=[1, 1, 4, 168960],
        output_dtype=ttnn.bfloat16,
    )

    # Prepare host inputs
    host_inputs = []
    for _ in range(iterations):
        _, input_tensor = create_unet_input_tensors(batch, groups, channel_order="first", pad=False, fold=True)
        host_inputs.append(input_tensor)

    # Run pipeline with optimized path
    logger.info("Compiling model and recording trace...")
    pipe.compile(host_inputs[0])

    pipe.preallocate_output_tensors_on_host(
        iterations,
        output_shape=[1, 1, 4, 168960],
        output_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        dram_output_memory_config=dram_output_memory_config,
    )

    logger.info(f"Running model for {iterations} iterations using optimized fast path")
    start = time.time()
    outputs = pipe.enqueue_fast(host_inputs).pop_all_fast()
    end = time.time()
    inference_time = (end - start) / iterations
    logger.info(f"OPTIMIZED - Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"OPTIMIZED - Average model performance={iterations * groups * batch / (end-start) : .2f} fps")

    pipe.cleanup()

    # Verify outputs
    assert len(outputs) == len(host_inputs)
    B, C, H, W = torch_output_tensor.shape
    for i in range(len(outputs)):
        verify_with_pcc(
            torch_output_tensor,
            ttnn.to_torch(outputs[i]).reshape(B, C, H, W),
            pcc=UNET_FULL_MODEL_PCC if is_wormhole_b0(device) else UNET_FULL_MODEL_PCC_BH,
        )
    logger.info("PCC check passed.")
