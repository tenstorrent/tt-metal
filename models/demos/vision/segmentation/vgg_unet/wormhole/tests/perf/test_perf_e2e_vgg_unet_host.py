# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Non-trace HOST end-to-end perf test. Iteration 0 is compile + program-cache miss; later iterations
# are cache hits. A descriptor rebuild-on-cache-hit regression (#48928) balloons the steady-state
# cache-hit host time, so this guards against it.

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.vision.segmentation.vgg_unet.common.reference.vgg_unet import UNetVGG19
from models.demos.vision.segmentation.vgg_unet.common.ttnn.model_preprocessing import create_vgg_unet_model_parameters
from models.demos.vision.segmentation.vgg_unet.common.ttnn.ttnn_vgg_unet import Tt_vgg_unet
from models.perf.perf_utils import prep_perf_report


def get_expected_times():
    # (compile+miss, steady-state cache-hit). Calibrated on wormhole_b0.
    return (
        45.0,
        0.012,
    )  # wh_b0: measured compile ~33s, cache-hit ~0.009s (Move x52/Concat x12 still rebuild pre-fix); tighten once bound


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_perf_e2e_vgg_unet(device, batch_size, reset_seeds):
    torch_input = torch.randn((batch_size, 3, 256, 256))
    torch_model = UNetVGG19()
    torch_model.eval()

    parameters = create_vgg_unet_model_parameters(torch_model, torch_input, device)
    ttnn_model = Tt_vgg_unet(device, parameters, parameters.conv_args)

    iterations = 4
    durations = []
    for _ in range(iterations):
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16).to(device, ttnn.L1_MEMORY_CONFIG)
        start = time.time()
        output = ttnn_model(ttnn_input)
        output = ttnn.from_device(output)
        durations.append(time.time() - start)

    inference_and_compile_time = durations[0]
    inference_time = min(durations[1:])  # steady-state cache-hit

    expected_compile_time, expected_inference_time = get_expected_times()
    prep_perf_report(
        model_name="VGG_UNet",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="non-trace host e2e",
        inference_time_cpu=0.0,
    )
    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Cache-hit inference time (avg): {inference_time}")
