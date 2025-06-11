# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
from models.demos.vgg_unet.ttnn.model_preprocessing import create_vgg_unet_model_parameters
from models.demos.vgg_unet.ttnn.ttnn_vgg_unet import Tt_vgg_unet
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import profiler, run_for_wormhole_b0


def get_expected_times(name):
    base = {"vgg_unet": (31.6, 0.15)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@run_for_wormhole_b0()
# Using random weights as pretrained ones aren't available in open-source; real weights were manually trained and uploaded to Drive
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        # True,
    ],
    ids=[
        "pretrained_weight_false",
        # "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True, ids=["0"])
def test_vgg_unet(device, reset_seeds, model_location_generator, use_pretrained_weight):
    torch.manual_seed(0)

    # Input creation
    torch_input = torch.randn((1, 3, 256, 256), dtype=torch.bfloat16)
    torch_input = torch_input.float()
    batch_size = torch_input.shape[0]

    # Model initialisation
    torch_model = UNetVGG19()

    # Pre-trained weights processing
    if use_pretrained_weight:
        weights_pth = "models/experimental/vgg_unet/vgg_unet_torch.pth"
        torch_dict = torch.load(weights_pth)
        new_state_dict = dict(zip(torch_model.state_dict().keys(), torch_dict.values()))
        torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    # Model call
    ref = torch_model(torch_input)

    # Weights pre-processing
    parameters = create_vgg_unet_model_parameters(torch_model, torch_input, device)
    ttnn_model = Tt_vgg_unet(device, parameters, parameters.conv_args)

    torch_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    result = ttnn_model(ttnn_input)
    ttnn.deallocate(result)

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    iterations = 16
    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        result = ttnn_model(ttnn_input)
        ttnn.deallocate(result)
        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation took {compile_time:.1f} s")
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(
        f"Mean inference time for {batch_size} (batch) images was {(mean_inference_time * 1000.0):.2f} ms ({batch_size / mean_inference_time:.2f} fps)"
    )

    expected_compile_time, expected_inference_time = get_expected_times("vgg_unet")

    prep_perf_report(
        model_name="models/experimental/vgg_unet",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")
    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
    logger.info("Exit vgg_unet perf test")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 92.738],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_vgg_unet(batch_size, expected_perf):
    subdir = "ttnn_vgg_unet"
    num_iterations = 1
    margin = 0.03
    command = f"pytest tests/ttnn/integration_tests/vgg_unet/test_vgg_unet.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_vgg_unet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
