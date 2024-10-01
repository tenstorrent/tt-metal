# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import time

from torchvision import models
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.vgg.tt import ttnn_vgg
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import is_grayskull


def get_expected_times(vgg):
    return (16, 10.5)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity", ((1, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),)
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 3, 224, 224),
    ],
)
@pytest.mark.parametrize(
    "model_class, weights",
    [
        (models.vgg11, models.VGG11_Weights.IMAGENET1K_V1),
        (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    ],
)
def test_vgg(
    device,
    input_shape,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_class,
    weights,
):
    disable_persistent_kernel_cache()
    torch_model = model_class(weights=weights)
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor_nchw = torch.rand(input_shape, dtype=torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_vgg.custom_preprocessor,
    )

    if model_class == models.vgg11:
        ttnn_model = ttnn_vgg.ttnn_vgg11
        model_name = "VGG11"
    else:
        ttnn_model = ttnn_vgg.ttnn_vgg16
        model_name = "VGG16"

    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weight_dtype,
        "ACTIVATIONS_DTYPE": act_dtype,
    }

    conv_config = ttnn.Conv2dConfig(
        dtype=model_config["ACTIVATIONS_DTYPE"],
        weights_dtype=model_config["WEIGHTS_DTYPE"],
        math_fidelity=model_config["MATH_FIDELITY"],
        activation="relu",
        deallocate_activation=True,
        input_channels_alignment=16,
        act_block_h_override=0,
        transpose_shards=True,
    )

    torch_batched_tensor = torch_input_tensor_nchw.repeat(batch_size, 1, 1, 1)
    torch_input_tensor = torch.permute(torch_batched_tensor, (0, 2, 3, 1))
    tt_batched_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    durations = []
    for i in range(2):
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        start = time.time()
        ttnn_output = ttnn_model(device, tt_batched_input_tensor, parameters, batch_size, model_config)
        output = ttnn.from_device(ttnn_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(model_name)
    prep_perf_report(
        model_name=model_name,
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
    "batch_size, model_name",
    [
        (1, "ttnn_vgg11"),
        (1, "ttnn_vgg16"),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_vgg(batch_size, model_name):
    subdir = model_name
    num_iterations = 1
    margin = 0.03

    if model_name == "ttnn_vgg11":
        expected_perf = 79.3 if is_grayskull() else 105.7
        command = f"pytest tests/ttnn/integration_tests/vgg/test_ttnn_vgg11.py"
    else:
        expected_perf = 73.6 if is_grayskull() else 92.6
        command = f"pytest tests/ttnn/integration_tests/vgg/test_ttnn_vgg16.py"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
