# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.efficientnetb0.runner.performant_runner import EfficientNetb0PerformantRunner
from models.utility_functions import run_for_wormhole_b0
from models.experimental.efficientnetb0.tt.model_preprocessing import get_mesh_mappers


def run_efficientnetb0_inference(
    device,
    batch_size_per_device,
    resolution,
    act_dtype=ttnn.bfloat16,
    weight_dtype=ttnn.bfloat16,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)
    performant_runner = EfficientNetb0PerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
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
        f"Model: ttnn_efficientnetb0 - batch_size: {batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size / inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (224, 224),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    resolution,
):
    run_efficientnetb0_inference(
        device,
        batch_size,
        resolution,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (224, 224),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant_dp(
    mesh_device,
    batch_size,
    act_dtype,
    weight_dtype,
    resolution,
):
    run_efficientnetb0_inference(
        mesh_device,
        batch_size,
        resolution,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
    )
