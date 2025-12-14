# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE


def run_e2e_performant_bge(device, batch_size, sequence_length, act_dtype, weight_dtype, model_location_generator):
    """Run end-to-end performance test for BGE-large-en-v1.5 model."""
    performant_runner = BGEPerformantRunner(
        device=device,
        device_batch_size=batch_size,
        sequence_length=sequence_length,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
        model_location_generator=model_location_generator,
        model_name="BAAI/bge-large-en-v1.5",
    )
    performant_runner._capture_bge_trace_2cqs()

    inference_iter_count = 50
    t0 = time.time()
    for _ in range(inference_iter_count):
        _ = performant_runner.run()
    ttnn.synchronize_device(device)
    t1 = time.time()
    performant_runner.release()

    inference_time_avg = round((t1 - t0) / inference_iter_count, 6)
    sentences_per_sec = round(batch_size * device.get_num_devices() / inference_time_avg)

    logger.info(f"BGE-Large-EN-v1.5 Performance:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  One inference iteration time (sec): {inference_time_avg}")
    logger.info(f"  Sentences per second: {sentences_per_sec}")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "act_dtype, weight_dtype",
    ((ttnn.bfloat16, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("batch_size, sequence_length", [(8, 512)])
def test_e2e_performant_bge(device, batch_size, sequence_length, act_dtype, weight_dtype, model_location_generator):
    """Test single device performance for BGE-large-en-v1.5."""
    return run_e2e_performant_bge(
        device, batch_size, sequence_length, act_dtype, weight_dtype, model_location_generator
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "act_dtype, weight_dtype",
    ((ttnn.bfloat16, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("device_batch_size, sequence_length", [(8, 512)])
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant_bge_dp(
    mesh_device, device_batch_size, sequence_length, act_dtype, weight_dtype, model_location_generator
):
    """Test multi-device (data parallel) performance for BGE-large-en-v1.5."""
    return run_e2e_performant_bge(
        mesh_device, device_batch_size, sequence_length, act_dtype, weight_dtype, model_location_generator
    )
