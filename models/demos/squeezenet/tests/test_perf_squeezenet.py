# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
import ttnn
import torch.nn as nn
import time
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache, profiler
from models.demos.squeezenet.tt.tt_squeezenet import tt_squeezenet
from models.utility_functions import is_grayskull, is_wormhole_b0
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.perf.perf_utils import prep_perf_report


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = ttnn.from_torch(model.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)

    return parameters


def get_expected_times(tt_squeezenet):
    return {tt_squeezenet: (0.46, 36.5) if is_grayskull() else (0.39, 55)}[tt_squeezenet]


@pytest.mark.parametrize(
    "batch_size",
    ([8]),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
def test_perf_squeezenet(device, batch_size):
    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    torch_squeezenet.eval()
    disable_persistent_kernel_cache()
    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_squeezenet, custom_preprocessor=custom_preprocessor, device=None
    )
    profiler.end(f"preprocessing_parameter")
    filename = "models/demos/squeezenet/demo/dog_image.jpeg"
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.repeat(batch_size, 1, 1, 1)
    cpu_key = "ref_key"
    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = torch_squeezenet(input_batch)
        profiler.end(cpu_key)
        durations = []
        for _ in range(2):
            profiler.start(f"preprocessing_input")
            tt_input = ttnn.from_torch(torch.permute(input_batch, (0, 2, 3, 1)))
            profiler.start(f"preprocessing_input")
            start = time.time()
            tt_out = tt_squeezenet(device=device, parameters=parameters, tt_input=tt_input)
            tt_out = ttnn.to_torch(tt_out)
            end = time.time()
            durations.append(end - start)
            enable_persistent_kernel_cache()
    inference_and_compile_time, *inference_times = durations
    average_inference_time = sum(inference_times) / len(inference_times)
    expected_inference_time, expected_compile_time = get_expected_times(tt_squeezenet)
    prep_perf_report(
        model_name="tt_squeezenet",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=average_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )
    logger.info(f"Compile time: {inference_and_compile_time - average_inference_time}")
    logger.info(f"Avg Inference time: {average_inference_time}")
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Sample(s) per second: {1 / average_inference_time * batch_size}")
    assert (
        inference_times[0] < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_times}"
    logger.info("Exit SqueezeNet perf test")


@pytest.mark.parametrize(
    "batch_size",
    [8],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal(batch_size, reset_seeds):
    subdir = "tt_squeezenet"
    num_iterations = 1
    margin = 0.03
    if is_grayskull():
        expected_perf = 929.5
    elif is_wormhole_b0():
        expected_perf = 1071.2

    command = f"pytest tests/ttnn/integration_tests/squeezenet/test_squeezenet.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"tt_squeezenet_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
