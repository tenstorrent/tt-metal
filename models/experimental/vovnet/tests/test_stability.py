# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.vovnet.common import VOVNET_L1_SMALL_SIZE
from models.experimental.vovnet.runner.performant_runner import VovnetPerformantRunner


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VOVNET_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
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
@pytest.mark.parametrize("test_duration", [5])
@pytest.mark.parametrize("pcc_check_interval", [5])
def test_vovnet_stability(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    test_duration,
    pcc_check_interval,
):
    performant_runner = VovnetPerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )

    # Initialize trace capture for VovNet
    performant_runner._capture_vovnet_trace_2cqs()

    logger.info(f"Running stability test for VovNet with resolution: {resolution} and batch size: {batch_size}")

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
            output = performant_runner.run(torch_input_tensor, check_pcc=False)

            if check_pcc:
                # Compute fresh torch reference output for the current input
                torch_output = performant_runner.runner_infra.torch_model(torch_input_tensor)
                # Convert ttnn output to torch for comparison
                ttnn_output_torch = ttnn.to_torch(
                    output, mesh_composer=performant_runner.runner_infra.output_mesh_composer
                )
                # Validate with proper PCC check
                performant_runner.runner_infra.validate(output_tensor=output, torch_output_tensor=torch_output)
                check_pcc = False

    performant_runner.release()
