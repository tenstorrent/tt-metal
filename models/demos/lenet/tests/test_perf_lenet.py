# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
import ttnn

from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.utility_functions import is_grayskull, is_wormhole_b0
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.perf.perf_utils import prep_perf_report
from models.demos.lenet import lenet_utils
from models.demos.lenet.tt import tt_lenet


def get_expected_times(tt_lenet):
    if is_grayskull():
        return {
            tt_lenet: (7.62, 0.05),
        }[tt_lenet]
    elif is_wormhole_b0():
        return {
            tt_lenet: (10.75, 0.049),
        }[tt_lenet]


@pytest.mark.parametrize(
    "batch_size",
    [64],
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
    torch_lenet, state_dict = lenet_utils.load_torch_lenet(pt_model_path, num_classes)
    model = torch_lenet.float()
    disable_persistent_kernel_cache()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=lenet_utils.custom_preprocessor,
    )
    parameters = lenet_utils.custom_preprocessor_device(parameters, device)
    x = test_input.permute(0, 2, 3, 1)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
    durations = []

    for _ in range(100):
        start = time.time()
        ttnn_output = tt_lenet.lenet(
            device=device,
            input_tensor=x,
            parameters=parameters,
        )
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, *inference_times = durations
    inference_time = sum(inference_times) / len(inference_times)
    expected_compile_time, expected_inference_time = get_expected_times(tt_lenet)

    prep_perf_report(
        model_name="tt_lenet",
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
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Sample(s) per second: {1 / inference_time * batch_size}")
    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
    logger.info("Exit Lenet perf test")


@pytest.mark.parametrize(
    "batch_size",
    [64],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal(batch_size, reset_seeds):
    subdir = "tt_lenet"
    num_iterations = 1
    margin = 0.03
    if is_grayskull():
        expected_perf = 83102.20
    elif is_wormhole_b0():
        expected_perf = 46313.985

    command = f"pytest tests/ttnn/integration_tests/lenet/test_lenet.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"tt_lenet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
