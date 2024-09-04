# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from loguru import logger
from transformers import AutoImageProcessor, ViTForImageClassification

import ttnn

from models.experimental.vit.tt.modeling_vit import vit_for_image_classification
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    torch_to_tt_tensor_rm,
    Profiler,
)
from models.perf.perf_utils import prep_perf_report


BATCH_SIZE = 1


def run_perf_vit(expected_inference_time, expected_compile_time, hf_cat_image_sample_input, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    image = hf_cat_image_sample_input

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    HF_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")  # loaded for the labels
    inputs = image_processor(image, return_tensors="pt")

    tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)

    tt_inputs = tt_inputs.to(device, ttnn.L1_MEMORY_CONFIG)
    tt_model = vit_for_image_classification(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = HF_model(**inputs).logits
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_inputs)[0]
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_inputs)[0]
        ttnn.synchronize_device(device)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_perf_report(
        model_name="vit",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="base-patch16",
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time
    logger.info(f"vit inference time: {second_iter_time}")
    logger.info(f"vit compile time: {compile_time}")


@pytest.mark.skip(reason="#7527: Test needs review")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            2.1,
            16,
        ),
    ),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
):
    run_perf_vit(
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )


@pytest.mark.skip(reason="#7527: Test needs review")
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            2.9,
            17.5,
        ),
    ),
)
def test_perf_virtual_machine(
    device,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
):
    run_perf_vit(
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )
