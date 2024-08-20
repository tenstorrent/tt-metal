# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from loguru import logger
import pytest
import ttnn

from models.experimental.deit.tt.deit_for_image_classification_with_teacher import (
    deit_for_image_classification_with_teacher,
)
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    torch_to_tt_tensor_rm,
    profiler,
)
from models.perf.perf_utils import prep_perf_report


BATCH_SIZE = 1


def run_perf_deit(expected_inference_time, expected_compile_time, hf_cat_image_sample_input, device):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    comments = "distilled-patch16-wteacher"

    image = hf_cat_image_sample_input

    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    HF_model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
    inputs = image_processor(image, return_tensors="pt")

    tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)
    tt_model = deit_for_image_classification_with_teacher(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = HF_model(**inputs).logits
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_inputs)[0]
        ttnn.synchronize_device(device)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_inputs)[0]
        ttnn.synchronize_device(device)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_perf_report(
        model_name="deit",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"deit {comments} inference time: {second_iter_time}")
    logger.info(f"deit {comments} compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            1.8,
            18,
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
    run_perf_deit(
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            2.0,
            19.5,
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
    run_perf_deit(
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )
