# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov12x.common import YOLOV12_L1_SMALL_SIZE
from models.demos.yolov12x.runner.performant_runner import YOLOv12xPerformantRunner
from models.demos.yolov12x.tt.common import get_mesh_mappers
from models.utility_functions import run_for_wormhole_b0


def run_yolov12x_inference(
    device,
    batch_size_per_device,
    resolution,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)
    performant_runner = YOLOv12xPerformantRunner(
        device,
        batch_size_per_device,
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
        f"ttnn_yolov12x_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round( batch_size / inference_time_avg)}"
    )


@pytest.mark.parametrize(
    "batch_size_per_device, resolution",
    [
        (
            1,
            (640, 640),
        ),
    ],
)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV12_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_performant(
    device,
    batch_size_per_device,
    resolution,
):
    run_yolov12x_inference(
        device,
        batch_size_per_device,
        resolution,
    )


@pytest.mark.parametrize(
    "batch_size_per_device, resolution",
    [
        (
            1,
            (640, 640),
        ),
    ],
)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV12_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_performant_dp(
    mesh_device,
    batch_size_per_device,
    resolution,
):
    run_yolov12x_inference(
        mesh_device,
        batch_size_per_device,
        resolution,
    )
