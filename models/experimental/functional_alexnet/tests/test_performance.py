# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import pytest
import torch
from models.utility_functions import is_grayskull
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from loguru import logger
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)

from torchvision import models
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_alexnet.tt.ttnn_alexnet import custom_preprocessor
from models.experimental.functional_alexnet.tt.ttnn_alexnet import ttnn_alexnet
from models.perf.perf_utils import prep_perf_report
from models.experimental.functional_alexnet.reference.alexnet import AlexNet


def get_expected_times(alexnet):
    return (4.74, 0.95)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity", ((1, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),)
)
@pytest.mark.parametrize("input_tensor", [torch.rand((10, 3, 64, 64))], ids=["input_tensor"])
def test_alexnet(device, input_tensor, batch_size, act_dtype, weight_dtype, math_fidelity):
    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: True,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    durations = []
    for i in range(2):
        start = time.time()
        ttnn_output_tensor = ttnn_alexnet(device, ttnn_input, parameters)
        tnn_output_tensor = ttnn.from_device(ttnn_output_tensor)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("alexnet")

    prep_perf_report(
        model_name="alexnet",
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


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 32.8],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_alexnet(batch_size, expected_perf):
    subdir = "ttnn_alexnet"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_grayskull() else 2705.5

    command = f"pytest tests/ttnn/integration_tests/alexnet/test_alexnet.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_alexnet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
