# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import torch
from loguru import logger
import ttnn

from models.experimental.efficientnetb0.reference import efficientnetb0
from models.experimental.efficientnetb0.tt.model_preprocessing import (
    create_efficientnetb0_input_tensors,
    create_efficientnetb0_model_parameters,
)
from models.experimental.efficientnetb0.tt import efficientnetb0 as ttnn_efficientnetb0
from efficientnet_pytorch import EfficientNet
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    run_for_wormhole_b0,
)


def get_expected_times(name):
    base = {"EfficientNetb0": (119, 0.052)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 7 * 1024}], indirect=True)
def test_perf(device, reset_seeds):
    disable_persistent_kernel_cache()

    model = EfficientNet.from_pretrained("efficientnet-b0").eval()

    state_dict = model.state_dict()
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = efficientnetb0.Efficientnetb0()

    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input, ttnn_input = create_efficientnetb0_input_tensors(device)
    batch_size = torch_input.shape[0]
    torch_output = torch_model(torch_input)
    conv_params, parameters = create_efficientnetb0_model_parameters(torch_model, torch_input, device=device)

    ttnn_model = ttnn_efficientnetb0.Efficientnetb0(device, parameters, conv_params)

    durations = []

    for i in range(2):
        start = time.time()
        torch_input, ttnn_input = create_efficientnetb0_input_tensors(device)
        ttnn_model_output = ttnn_model(ttnn_input)[0]
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("EfficientNetb0")

    prep_perf_report(
        model_name="models/experimental/EfficientNetb0",
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


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 100.5],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_efficientnetb0(batch_size, expected_perf):
    subdir = "ttnn_efficientnetb0"
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/efficientnetb0/test_ttnn_efficientnetb0.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_efficientnetb0{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
