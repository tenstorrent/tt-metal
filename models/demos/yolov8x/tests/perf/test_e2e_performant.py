# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

from models.demos.yolov8x.runner.performant_runner import YOLOv8xPerformantRunner
from models.demos.yolov8x.tt.ttnn_yolov8x_utils import get_mesh_mappers
from models.utility_functions import run_for_wormhole_b0


def run_yolov8x_trace_2cqs_inference(
    device,
    batch_size_per_device,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    performant_runner = YOLOv8xPerformantRunner(
        device,
        batch_size,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        outputs_mesh_composer=outputs_mesh_composer,
    )

    input_shape = (batch_size, 3, 640, 640)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        t0 = time.time()
        output = performant_runner.run(torch_input_tensor)
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    performant_runner.release()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_yolov8x_640x640_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size * num_devices/inference_time_avg)}"
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
def test_run_yolov8x_performant(
    device,
    batch_size_per_device,
):
    run_yolov8x_trace_2cqs_inference(
        device,
        batch_size_per_device,
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
def test_run_yolov8x_performant_dp(
    mesh_device,
    batch_size_per_device,
):
    run_yolov8x_trace_2cqs_inference(
        mesh_device,
        batch_size_per_device,
    )
