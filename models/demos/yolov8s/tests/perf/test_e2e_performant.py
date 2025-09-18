# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE
from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner


def run_yolov8s(
    device,
    batch_size_per_device,
    model_location_generator,
):
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    performant_runner = YOLOv8sPerformantRunner(
        device,
        batch_size,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
        model_location_generator=model_location_generator,
    )

    input_shape = (batch_size, 3, 640, 640)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    t0 = time.time()
    for _ in range(10):
        _ = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()
    inference_time_avg = round((t1 - t0) / 10, 6)
    logger.info(
        f"Model: ttnn_yolov8s - batch_size: {batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round((batch_size) / inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8s_trace_2cqs_inference(
    device,
    batch_size_per_device,
    model_location_generator,
):
    run_yolov8s(
        device,
        batch_size_per_device,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8s_trace_2cqs_dp_inference(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
):
    run_yolov8s(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
    )
