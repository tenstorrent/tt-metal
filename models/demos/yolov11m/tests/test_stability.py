# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.yolov11m.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11m.runner.performant_runner import YOLOv11PerformantRunner


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.parametrize("test_duration", [300])
@pytest.mark.parametrize("pcc_check_interval", [10])
def test_yolov11m_stability(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    test_duration,
    pcc_check_interval,
):
    performant_runner = YOLOv11PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=None,
    )

    logger.info(f"Running stability test for YOLOv11m with resolution: {resolution} and batch size: {batch_size}")

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

            torch_input_tensor = torch.randn((1, 3, *resolution), dtype=torch.float32)
            _ = performant_runner.run(torch_input_tensor, check_pcc=check_pcc)
            check_pcc = False

    performant_runner.release()
