# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.yolov9c.common import YOLOV9C_L1_SMALL_SIZE
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner
from models.demos.yolov9c.tt.model_preprocessing import get_mesh_mappers


def run_yolov9c_inference(
    model_location_generator,
    device,
    batch_size_per_device,
    resolution,
    model_task,
    act_dtype=ttnn.bfloat8_b,
    weight_dtype=ttnn.bfloat8_b,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    performant_runner = YOLOv9PerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_task=model_task,
        model_location_generator=model_location_generator,
        resolution=resolution,
        mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        mesh_composer=outputs_mesh_composer,
    )

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

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
        f"ttnn_yolov9 {model_task} - batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size / inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
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
def test_e2e_performant(
    model_location_generator,
    device,
    batch_size,
    resolution,
):
    run_yolov9c_inference(
        model_location_generator,
        device,
        batch_size,
        resolution,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
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
def test_e2e_performant_dp(
    model_location_generator,
    mesh_device,
    batch_size,
    resolution,
):
    run_yolov9c_inference(
        model_location_generator,
        mesh_device,
        batch_size,
        resolution,
        model_task="detect",
    )
