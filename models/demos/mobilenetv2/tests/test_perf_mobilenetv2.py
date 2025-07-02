# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.mobilenetv2.tt.model_preprocessing import create_mobilenetv2_model_parameters
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache, profiler
from tests.ttnn.integration_tests.mobilenetv2.test_mobilenetv2 import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE


def get_expected_times(name):
    base = {"mobilenetv2": (63.3, 0.14)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 224, 224, 3))], ids=["input_tensor"])
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_mobilenetv2(device, input_tensor, use_pretrained_weight, reset_seeds):
    # Check if weights file exists, if not, download them
    disable_persistent_kernel_cache()
    profiler.clear()
    batch_size = input_tensor.shape[0]
    weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"
    if not os.path.exists(weights_path):
        os.system("bash models/demos/mobilenetv2/weights_download.sh")
    if use_pretrained_weight:
        state_dict = torch.load(weights_path)
        ds_state_dict = {k: v for k, v in state_dict.items()}

        torch_model = Mobilenetv2()
        new_state_dict = {
            name1: parameter2
            for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
            if isinstance(parameter2, torch.FloatTensor)
        }
        torch_model.load_state_dict(new_state_dict)
    else:
        torch_model = Mobilenetv2()
        state_dict = torch_model.state_dict()

    torch_model.eval()

    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size)

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    ttnn_output_tensor = ttnn_model(ttnn_input)

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    iterations = 16
    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        ttnn_output_tensor = ttnn_model(ttnn_input)

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

    expected_compile_time, expected_inference_time = get_expected_times("mobilenetv2")

    prep_perf_report(
        model_name="models/demos/functional_mobilenetv2",
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
    logger.info("Exit mobilenetv2 perf test")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [MOBILENETV2_BATCH_SIZE, 3178],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_mobilenetv2(batch_size, expected_perf):
    subdir = "ttnn_mobilenetv2"
    num_iterations = 1
    margin = 0.03
    command = f"pytest tests/ttnn/integration_tests/mobilenetv2/test_mobilenetv2.py::test_mobilenetv2"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_mobilenetv2{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
