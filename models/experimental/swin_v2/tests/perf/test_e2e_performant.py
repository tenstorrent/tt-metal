# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from loguru import logger
import ttnn
from models.experimental.swin_v2.runner.performant_runner import SwinV2PerformantRunner
from models.utility_functions import run_for_wormhole_b0
from models.experimental.swin_v2.common import SWIN_V2_L1_SMALL_SIZE


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
    [{"l1_small_size": SWIN_V2_L1_SMALL_SIZE, "trace_region_size": 18776064, "num_command_queues": 2}],
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
    performant_runner = SwinV2PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )
    performant_runner._capture_swinv2_trace_2cqs()

    inference_times = []
    for _ in range(10):
        input_shape = (1, 3, 512, 512)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        t0 = time.time()
        _ = performant_runner.run(torch_input_tensor)
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_swin_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
