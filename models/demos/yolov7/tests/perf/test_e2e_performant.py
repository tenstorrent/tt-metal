# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import sys
import time

import pytest
import torch
from loguru import logger

import models.demos.yolov7.reference.yolov7_model as yolov7_model
import models.demos.yolov7.reference.yolov7_utils as yolov7_utils
import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov7.common import YOLOV7_L1_SMALL_SIZE
from models.demos.yolov7.runner.performant_runner import YOLOv7PerformantRunner
from models.utility_functions import run_for_wormhole_b0

sys.modules["models.common"] = yolov7_utils
sys.modules["models.yolo"] = yolov7_model


def run_yolov7_trace_2cqs_inference(
    model_location_generator,
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    performant_runner = YOLOv7PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        outputs_mesh_composer=outputs_mesh_composer,
    )

    input_shape = (batch_size, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    t0 = time.time()
    for _ in range(10):
        _ = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()

    inference_time_avg = round((t1 - t0) / 10, 6)
    logger.info(
        f"ttnn_yolov7_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size / inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV7_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.models_performance_bare_metal
def test_e2e_performant(
    model_location_generator,
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
):
    run_yolov7_trace_2cqs_inference(
        model_location_generator,
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV7_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant_dp(
    model_location_generator,
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
):
    run_yolov7_trace_2cqs_inference(
        model_location_generator,
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution,
    )
