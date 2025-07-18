# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, resolution",
    [
        (
            1,
            (640, 640),
        ),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant(
    device,
    model_location_generator,
    batch_size,
    resolution,
    act_dtype=ttnn.bfloat8_b,
    weight_dtype=ttnn.bfloat8_b,
):
    performant_runner = YOLOv9PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_task="detect",
        resolution=resolution,
        model_location_generator=None,
    )
    performant_runner._capture_yolov9_trace_2cqs()
    input_shape = (1, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    inference_times = []
    for _ in range(10):
        t0 = time.time()
        out = performant_runner.run(torch_input_tensor)
        output = ttnn.to_torch(out[0], dtype=torch.float32)
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_yolov9 Detect - batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
