# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger
from tqdm import tqdm

from models.common.utility_functions import run_for_wormhole_b0
from models.demos.yolov12x.common import YOLOV12_L1_SMALL_SIZE
from models.demos.yolov12x.runner.performant_runner import YOLOv12xPerformantRunner
from models.demos.yolov12x.tt.common import get_mesh_mappers


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV12_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.parametrize("test_duration", [300])
@pytest.mark.parametrize("pcc_check_interval", [5])
def test_yolov12x_stability(
    device,
    batch_size,
    model_location_generator,
    resolution,
    test_duration,
    pcc_check_interval,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)
    performant_runner = YOLOv12xPerformantRunner(
        device,
        device_batch_size=batch_size,
        resolution=resolution,
        mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        mesh_composer=outputs_mesh_composer,
        model_location_generator=model_location_generator,
    )

    logger.info(f"Running stability test for YOLOv12x with resolution: {resolution} and batch size: {batch_size}")

    num_devices = device.get_num_devices()
    total_batch_size = batch_size * num_devices

    pcc_iter = 0
    check_pcc = False
    start_time = time.time()

    with tqdm(total=test_duration, desc="Executing on device", unit="sec", mininterval=1) as pbar:
        while True:
            elapsed_time = round(time.time() - start_time, 1)
            pbar.update(min(elapsed_time, test_duration) - pbar.n)

            if elapsed_time >= test_duration:
                break

            if elapsed_time >= pcc_iter * pcc_check_interval:
                check_pcc = True
                pcc_iter += 1

            torch_input_tensor = torch.randn((total_batch_size, 3, *resolution), dtype=torch.float32)
            _ = performant_runner.run(torch_input_tensor, check_pcc=check_pcc)
            check_pcc = False

    performant_runner.release()
