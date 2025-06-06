# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",  # To run the test for instance segmentation
        # "detect",  # Uncomment to run the test for Object Detection
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    model_task,
    model_location_generator,
    resolution,
):
    performant_runner = YOLOv9PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_task=model_task,
        resolution=resolution,
        model_location_generator=None,
    )
    performant_runner._capture_yolov9_trace_2cqs()
    input_shape = (1, *resolution, 3)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_input_tensor = F.pad(torch_input_tensor, (0, 29), mode="constant", value=0)
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    inference_times = []
    for _ in range(10):
        t0 = time.time()
        _ = performant_runner._execute_yolov9_trace_2cqs_inference(tt_inputs_host)
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_yolov9_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
