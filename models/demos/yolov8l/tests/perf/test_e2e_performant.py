# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8l.common import YOLOV8L_L1_SMALL_SIZE
from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def run_yolov8l(
    device,
    batch_size_per_device,
    model_location_generator,
):
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    performant_runner = YOLOv8lPerformantRunner(
        device,
        batch_size,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
        model_location_generator=model_location_generator,
    )

    input_shape = (batch_size, 3, 640, 640)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    if use_signpost:
        signpost(header="start")

    t0 = time.time()
    for _ in range(10):
        _ = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    if use_signpost:
        signpost(header="stop")

    performant_runner.release()
    inference_time_avg = round((t1 - t0) / 10, 6)
    logger.info(
        f"Model: ttnn_yolov8l - batch_size: {batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round((batch_size) / inference_time_avg)}"
    )


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8L_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8l_trace_2cqs_inference(
    device,
    batch_size_per_device,
    model_location_generator,
):
    run_yolov8l(
        device,
        batch_size_per_device,
        model_location_generator,
    )


@run_for_blackhole()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8L_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8l_trace_2cqs_dp_inference(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
):
    run_yolov8l(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
    )
