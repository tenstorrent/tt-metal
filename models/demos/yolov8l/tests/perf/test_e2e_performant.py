# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8l.common import YOLOV8L_INPUT_H, YOLOV8L_INPUT_W, yolov8l_l1_small_size_for_res
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
    resolution=(YOLOV8L_INPUT_H, YOLOV8L_INPUT_W),
):
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    performant_runner = YOLOv8lPerformantRunner(
        device,
        batch_size,
        inp_h=resolution[0],
        inp_w=resolution[1],
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
        model_location_generator=model_location_generator,
    )

    input_shape = (batch_size, 3, resolution[0], resolution[1])
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


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": yolov8l_l1_small_size_for_res(1280, 1280),
            # "trace_region_size": YOLOV8L_TRACE_REGION_SIZE_E2E,
            "trace_region_size": 35000000,
            "num_command_queues": 2,
        }
    ],
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


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": yolov8l_l1_small_size_for_res(1280, 1280),
            "trace_region_size": 35000000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
@pytest.mark.parametrize(
    "resolution",
    [(640, 640), (1280, 1280)],
    ids=["640", "1280"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        # (2, 2),
        (1, 8),
    ],
    indirect=True,
    ids=["t3k_1x8"],
)
def test_run_yolov8l_trace_2cqs_dp_inference(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
    resolution,
):
    run_yolov8l(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
        resolution=resolution,
    )
