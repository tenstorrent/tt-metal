# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.demos.vgg_unet.runner.performant_runner import VggUnetTrace2CQ
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (256, 256),
    ],
)
@pytest.mark.parametrize("test_duration", [3600])
@pytest.mark.parametrize("pcc_check_interval", [10])
def test_vgg_unet_stability(
    device,
    model_location_generator,
    batch_size,
    resolution,
    test_duration,
    pcc_check_interval,
):
    # Initialize VGG UNet performant runner
    vgg_unet_trace_2cq = VggUnetTrace2CQ()
    vgg_unet_trace_2cq.initialize_vgg_unet_trace_2cqs_inference(
        device,
        model_location_generator=model_location_generator,
        device_batch_size=batch_size,
        use_pretrained_weight=False,
    )

    logger.info(f"Running stability test for VGG UNet with resolution: {resolution} and batch size: {batch_size}")

    pcc_iter = 0
    check_pcc = True
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

            torch_input_tensor = torch.randn((batch_size, 3, *resolution), dtype=torch.float32)
            tt_output = vgg_unet_trace_2cq.run(torch_input_tensor)
            if check_pcc:
                with torch.no_grad():
                    reference_output = vgg_unet_trace_2cq.test_infra.torch_model(torch_input_tensor)
                ttnn_output = ttnn.to_torch(tt_output, mesh_composer=vgg_unet_trace_2cq.test_infra.mesh_composer)
                ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
                ttnn_output = ttnn_output.reshape(reference_output.shape)
                pcc_passed, pcc_message = assert_with_pcc(reference_output, ttnn_output, pcc=0.98)
                logger.info(f"VGG_Unet, PCC={pcc_message}")
                assert pcc_passed, pcc_message
            check_pcc = False

    vgg_unet_trace_2cq.release_vgg_unet_trace_2cqs_inference()
