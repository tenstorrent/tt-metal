# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from models.demos.ufld_v2.ttnn.ttnn_ufld_v2 import TtnnUFLDv2
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    is_wormhole_b0,
    run_for_wormhole_b0,
)
from tests.ttnn.integration_tests.ufld_v2.test_ttnn_ufld_v2 import custom_preprocessor_whole_model


def get_expected_times(name):
    base = {"ufld_v2": (36.6, 0.1)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        # True
    ],
    ids=[
        "pretrained_weight_false",
        # "pretrained_weight_true",
    ],
)
def test_ufld_v2_perf(device, batch_size, input_channels, height, width, use_pretrained_weight, min_channels=8):
    disable_persistent_kernel_cache()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width))
    reference_model = TuSimple34(input_height=height, input_width=width)
    if use_pretrained_weight:
        weights_path = "models/demos/ufld_v2/tusimple_res34.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/ufld_v2/weights_download.sh")
            state_dict = torch.load(weights_path)
            new_state_dict = {}
            for key, value in state_dict["model"].items():
                new_key = key.replace("model.", "res_model.")
            new_state_dict[new_key] = value
            reference_model.load_state_dict(new_state_dict)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor_whole_model,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=reference_model, run_model=lambda model: reference_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnUFLDv2(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    n, c, h, w = torch_input_tensor.shape
    if c == 3:  # for sharding config of padded input
        c = min_channels
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    durations = []
    for i in range(2):
        ttnn_input_tensor_sharded = ttnn_input_tensor.to(device, input_mem_config)
        start = time.time()
        ttnn_model_output = ttnn_model(ttnn_input_tensor_sharded, batch_size=batch_size)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("ufld_v2")

    prep_perf_report(
        model_name="models/demos/ufld_v2",
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


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf,test",
    [
        [1, 340, "UFLD-v2"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_ufld_v2(batch_size, expected_perf, test):
    subdir = "ttnn_UFLD_v2"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest tests/ttnn/integration_tests/ufld_v2/test_ttnn_ufld_v2.py::test_ufld_v2_model"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_ufld_v2{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
