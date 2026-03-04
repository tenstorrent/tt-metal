# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from transformers import AutoModelForDepthEstimation

import ttnn
from models.experimental.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor
from models.common.utility_functions import profiler
from models.perf.perf_utils import prep_perf_report


BATCH_SIZE = 1
MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


def run_perf_depth_anything_v2(expected_inference_time, expected_compile_time, device):
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    comments = "depth-anything-v2-large"

    # Load reference model
    torch_model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, trust_remote_code=True)
    torch_model.eval()

    # Create input
    input_shape = (BATCH_SIZE, 3, 518, 518)
    pixel_values = torch.randn(input_shape)

    # Run CPU reference
    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = torch_model(pixel_values).predicted_depth
        profiler.end(cpu_key)

    # Initialize TT model
    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)

    tt_pixel_values = ttnn.from_torch(
        pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    with torch.no_grad():
        # First iteration (includes compile)
        profiler.start(first_key)
        tt_output = tt_model(tt_pixel_values)
        ttnn.synchronize_device(device)
        profiler.end(first_key)

        # Second iteration (inference only)
        profiler.start(second_key)
        tt_output = tt_model(tt_pixel_values)
        ttnn.synchronize_device(device)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_perf_report(
        model_name="depth_anything_v2_large",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"depth_anything_v2 {comments} inference time: {second_iter_time}")
    logger.info(f"depth_anything_v2 {comments} compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.067,  # ~15 FPS target
            30,
        ),
    ),
)
def test_perf_bare_metal(
    device,
    expected_inference_time,
    expected_compile_time,
):
    run_perf_depth_anything_v2(
        expected_inference_time,
        expected_compile_time,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.1,
            35,
        ),
    ),
)
def test_perf_virtual_machine(
    device,
    expected_inference_time,
    expected_compile_time,
):
    run_perf_depth_anything_v2(
        expected_inference_time,
        expected_compile_time,
        device,
    )
