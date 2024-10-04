# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
import time

from models.demos.swin.tt import ttnn_optimized_swin
from transformers import SwinForImageClassification as HF_SwinForImageClassification
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import is_grayskull, is_wormhole_b0
from models.demos.swin.tt.swin_utils import get_relative_position, get_attn_mask


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("model_name", ["microsoft/swin-tiny-patch4-window7-224"])
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    ([8, 4, 50.00],),
)
def test_performance_swin(
    batch_size,
    model_name,
    expected_inference_time,
    expected_compile_time,
    device,
):
    hugging_face_reference_model = HF_SwinForImageClassification.from_pretrained(model_name)
    hugging_face_reference_model.eval()
    pixel_values = torch.rand(batch_size, 3, 224, 224)
    # set up tokenizer
    disable_persistent_kernel_cache()

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name="ttnn_optimized_swin",
        initialize_model=lambda: hugging_face_reference_model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    cpu_key = "ref_key"

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = hugging_face_reference_model(pixel_values)
        profiler.end(cpu_key)

        durations = []
        for _ in range(2):
            profiler.start(f"preprocessing_input")
            tt_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            bias_table = get_relative_position(hugging_face_reference_model.config, parameters.swin, device)
            attention_mask_list = get_attn_mask(hugging_face_reference_model.config, device)

            profiler.end(f"preprocessing_input")

            start = time.time()
            tt_output = ttnn_optimized_swin.swin_for_image_classification(
                hugging_face_reference_model.config,
                pixel_values=tt_pixel_values,
                parameters=parameters,
                device=device,
                bias_table=bias_table,
                attention_mask_list=attention_mask_list,
            )
            tt_output = ttnn.from_device(tt_output)
            end = time.time()

            durations.append(end - start)
            enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    prep_perf_report(
        model_name=f"ttnn_{model_name}_optimized",
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
    logger.info("Exit Swin perf test")


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test",
    [
        [8, "microsoft/swin-tiny-patch4-window7-224"],
    ],
)
def test_swin_perf_device(batch_size, test, reset_seeds):
    subdir = "ttnn_swin"
    margin = 0.03
    num_iterations = 1
    if is_grayskull():
        expected_perf = 26
    elif is_wormhole_b0():
        expected_perf = 39

    command = f"pytest tests/ttnn/integration_tests/swin/test_ttnn_swin.py::test_swin_for_image_classification"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_swin{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
