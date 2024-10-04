# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import time
from pathlib import Path

from torchvision import models
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.demos.lenet.tt import tt_lenet
from models.utility_functions import is_grayskull, is_wormhole_b0
from models.demos.lenet import lenet_utils
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.perf.perf_utils import prep_perf_report


def get_expected_times(tt_lenet):
    if is_grayskull():
        return {
            tt_lenet: (3.7, 0.7),
        }[tt_lenet]
    elif is_wormhole_b0():
        return {
            tt_lenet: (9.52, 0.91),
        }[tt_lenet]


@pytest.mark.parametrize(
    "batch_size",
    [8],
)
@pytest.mark.parametrize(
    "tt_lenet",
    [tt_lenet],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
def test_perf_lenet(device, batch_size, tt_lenet, model_location_generator, reset_seeds):
    num_classes = 10
    test_input, images, outputs = lenet_utils.get_test_data(batch_size)

    pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
    torch_LeNet, state_dict = lenet_utils.load_torch_lenet(pt_model_path, num_classes)
    model = torch_LeNet.float()
    model = model.eval()
    disable_persistent_kernel_cache()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_LeNet,
        custom_preprocessor=lenet_utils.custom_preprocessor,
    )
    parameters = lenet_utils.custom_preprocessor_device(parameters, device)

    x = test_input.permute(0, 2, 3, 1)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
    durations = []
    for _ in range(2):
        start = time.time()

        ttnn_output = tt_lenet.Lenet(
            device=device,
            model=model,
            input_tensor=x,
            batch_size=batch_size,
            parameters=parameters,
            num_classes=num_classes,
            reset_seeds=reset_seeds,
        )
        end = time.time()
        durations.append(end - start)

    inference_and_compile_time, *inference_times = durations
    average_inference_time = sum(inference_times) / len(inference_times)
    expected_compile_time, expected_inference_time = get_expected_times(tt_lenet)

    prep_perf_report(
        model_name="tt_lenet",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=average_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - average_inference_time}")
    logger.info(f"Inference time: {average_inference_time}")
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Sample(s) per second: {1 / average_inference_time * batch_size}")


@pytest.mark.parametrize(
    "batch_size",
    [8],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal(batch_size, reset_seeds):
    subdir = "tt_lenet"
    num_iterations = 1
    margin = 0.03
    if is_grayskull():
        expected_perf = 419.5
    elif is_wormhole_b0():
        expected_perf = 988.29

    command = f"pytest tests/ttnn/integration_tests/lenet/test_lenet.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"tt_lenet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
