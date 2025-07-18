# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

from models.demos.yolov8s.demo.demo_utils import get_mesh_mappers
from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner
from models.utility_functions import run_for_wormhole_b0


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
        batch_size_per_device,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )

    input_shape = (batch_size_per_device, 3, 640, 640)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    for i in range(batch_size - 1):
        torch_input_tensor = torch.cat([torch_input_tensor] * batch_size, dim=0)

    inference_times = []
    for _ in range(10):
        t0 = time.time()
        _ = performant_runner.run(torch_input_tensor)
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()
    inference_time_avg = sum(inference_times) / len(inference_times)
    print(f"num_devices: {num_devices} batch_size: {batch_size}")
    FPS = round((batch_size * num_devices) / inference_time_avg)
    logger.info(
        f"Model: ttnn_yolov8s - batch_size: {batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {FPS}"
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
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
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
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
