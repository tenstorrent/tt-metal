# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from loguru import logger
import ttnn
from models.experimental.swin_s.runner.performant_runner import SwinSPerformantRunner
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.swin_s.common import SWIN_S_L1_SMALL_SIZE


def run_e2e_performant(
    device, device_batch_size, act_dtype, weight_dtype, model_location_generator, resolution, channels=3
):
    batch_size = device_batch_size * device.get_num_devices()
    performant_runner = SwinSPerformantRunner(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )
    performant_runner._capture_swins_trace_2cqs()
    input_shape = (batch_size, channels, resolution[0], resolution[1])
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    inference_times = []
    iterations_count = 50
    for i in range(iterations_count):
        t0 = time.time()
        out = performant_runner.run(torch_input_tensor)
        if i == iterations_count - 1:
            ttnn.synchronize_device(device)
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_swin_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )


@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SWIN_S_L1_SMALL_SIZE, "trace_region_size": 16998400, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_performant(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    return run_e2e_performant(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SWIN_S_L1_SMALL_SIZE, "trace_region_size": 16998400, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_performant_dp(
    mesh_device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    return run_e2e_performant(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )
