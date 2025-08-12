# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.efficientnetb0.runner.performant_runner import EfficientNetb0PerformantRunner
from models.utility_functions import run_for_wormhole_b0
from models.experimental.efficientnetb0.common import EFFICIENTNETB0_L1_SMALL_SIZE


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": EFFICIENTNETB0_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (224, 224),
    ],
)
@pytest.mark.models_performance_bare_metal
def test_e2e_performant(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    performant_runner = EfficientNetb0PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )
    performant_runner._capture_efficientnetb0_trace_2cqs()
    input_shape = (1, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    t0 = time.time()
    for _ in range(10):
        out = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()

    inference_time_avg = round((t1 - t0) / 10, 6)
    logger.info(
        f"ttnn_efficientnetb0_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
