# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import time
import ttnn
import pytest
import torch
from loguru import logger
from tqdm import tqdm

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.retinanet.runner.performant_runner import RetinaNetPerformantRunner


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 16384, "trace_region_size": 1702912, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
@pytest.mark.parametrize("test_duration", [5])
@pytest.mark.parametrize("pcc_check_interval", [5])
def test_retinanet_stability(
    device,
    batch_size,
    model_location_generator,
    resolution,
    test_duration,
    pcc_check_interval,
):
    # Define model config
    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    }

    performant_runner = RetinaNetPerformantRunner(
        device,
        batch_size,
        resolution=resolution,
        model_location_generator=None,
        input_path="models/experimental/retinanet/resources/dog.jpg",
        model_config=model_config,
    )

    logger.info(f"Running stability test for RetinaNet with resolution: {resolution} and batch size: {batch_size}")

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
