# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.unet_3d.demo.config import load_config
from models.demos.unet_3d.runner.performant_runner import UNet3DRunner


def run_unet3d_trace_2cqs_inference(
    device,
):
    config = load_config()
    model = UNet3DRunner()
    model.initialize_inference(device, config)
    batch_size = model.test_infra.batch_size
    resolution = config["dataset"]["slice_builder"]["patch_shape"]
    channels = config["model"]["in_channels"]
    input_shape = (batch_size, channels, resolution[0], resolution[1], resolution[2])
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    inference_iter_count = 10
    t0 = time.time()
    for iter in range(0, inference_iter_count):
        _ = model.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    model.release_inference()
    inference_time_avg = round((t1 - t0) / inference_iter_count, 6)
    logger.info(
        f"ttnn_unet_3d. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size / inference_time_avg)}"
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 8192, "trace_region_size": 679936, "num_command_queues": 2}], indirect=True
)
def test_unet3d_e2e(device):
    run_unet3d_trace_2cqs_inference(device)


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 8192, "trace_region_size": 679936, "num_command_queues": 2}], indirect=True
)
def test_unet3d_e2e_dp(mesh_device):
    run_unet3d_trace_2cqs_inference(mesh_device)
