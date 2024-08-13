# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import time
import pytest

from models.demos.ttnn_resnet.tests.ttnn_resnet_test_infra import create_test_infra
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull
import ttnn
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report


@skip_for_wormhole_b0("This will be enabled after WH testing")
@skip_for_grayskull("#9168: Resnet50 performance test failing after removing 1x1s2 matmul fallback into conv")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [
            20,
            "batch_size=20-act_dtype=DataType.BFLOAT8_B-weight_dtype=DataType.BFLOAT8_B-math_fidelity=MathFidelity.LoFi-device_params=l1_small_size_24576",
            7363,
        ],
    ],
)
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    subdir = "ResNet50"
    num_iterations = 3
    margin = 0.03
    command = f"pytest tests/ttnn/integration_tests/resnet/test_ttnn_functional_resnet50.py::test_resnet_50[{test}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ResNet50-{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )


@skip_for_wormhole_b0("This will be enabled after WH testing")
@skip_for_grayskull("#9168: Resnet50 performance test failing after removing 1x1s2 matmul fallback into conv")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "model_name,batch_size,act_dtype,weight_dtype,math_fidelity,expected_compile_time,expected_inference_time",
    [("ResNet50", 20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, 15, 0.015)],
)
def test_performance(
    device,
    use_program_cache,
    model_name,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    expected_compile_time,
    expected_inference_time,
):
    disable_persistent_kernel_cache()

    num_iterations = 10

    test_infra = create_test_infra(device, batch_size, act_dtype, weight_dtype, math_fidelity)

    durations = []
    for _ in range(num_iterations):
        test_infra.preprocess_torch_input()
        start = time.time()
        tt_output = test_infra.run()
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, *inference_times = durations
    average_inference_time = sum(inference_times) / len(inference_times)

    prep_perf_report(
        model_name=f"{model_name}-{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=average_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    test_infra.validate()

    logger.info(f"Compile time: {inference_and_compile_time - average_inference_time}")
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Average Inference time: {average_inference_time}")
    logger.info(f"Samples per second: {1 / average_inference_time * batch_size}")
    logger.info(f"PCC: {test_infra.pcc_message}")
