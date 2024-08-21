# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np

from loguru import logger
from datasets import load_dataset
from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher

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
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.deit.tests.demo_utils import get_data


BATCH_SIZE = 1


def run_perf_deit(
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
    iterations,
    model_location_generator,
):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    cpu_key = "ref_key"
    comments = "distilled-patch16-wteacher"

    sample_image = hf_cat_image_sample_input

    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    HF_model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
    input = image_processor(sample_image, return_tensors="pt")

    tt_input = torch_to_tt_tensor_rm(input["pixel_values"], device, put_on_device=False)
    tt_model_with_teacher = deit_for_image_classification_with_teacher(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = HF_model(**input).logits
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model_with_teacher(tt_input)[0]
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model_with_teacher(tt_input)[0]
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_output

        iteration = 0
        logger.info("ImageNet-1k validation Dataset")
        if iterations <= 50:
            input_loc = str(model_location_generator("sample_data"))
        else:
            input_loc = str(model_location_generator("ImageNet_data"))
        image_examples = get_data(input_loc)
        reference_labels = []
        predicted_labels = []

        weka_is_on = True
        if len(image_examples) == 0:
            weka_is_on = False
            files_raw = iter(load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True))
            image_examples = []
            sample_count = BATCH_SIZE * iterations
            for _ in range(sample_count):
                image_examples.append(next(files_raw))

        profiler.start(third_key)
        while iteration < iterations:
            if weka_is_on:
                input_image = image_examples[iteration].image
            else:
                input_image = image_examples[iteration]["image"]
            if input_image.mode == "L":
                input_image = input_image.convert(mode="RGB")

            inputs = image_processor(images=input_image, return_tensors="pt")
            tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)

            tt_output_with_teacher = tt_model_with_teacher(tt_inputs)[0]
            tt_output_with_teacher = tt_to_torch_tensor(tt_output_with_teacher).squeeze(0)[:, 0, :]

            prediction = tt_output_with_teacher.argmax(-1).item()
            predicted_labels.append(prediction)

            if weka_is_on:
                reference_labels.append(image_examples[iteration].label)
            else:
                reference_labels.append(image_examples[iteration]["label"])

            iteration += 1

        predicted_labels = np.array(predicted_labels)
        reference_labels = np.array(reference_labels)
        accuracy = np.mean(predicted_labels == reference_labels)

        logger.info("Accuracy")
        logger.info(accuracy)
        ttnn.synchronize_device(device)
        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)

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
    logger.info(f"deit inference for {iterations} Samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            1.8,
            18,
            50,
        ),
    ),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    iterations,
    model_location_generator,
):
    run_perf_deit(
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        iterations,
        model_location_generator,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            2.0,
            19.5,
            50,
        ),
    ),
)
def test_perf_virtual_machine(
    device,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    iterations,
    model_location_generator,
):
    run_perf_deit(
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
        iterations,
        model_location_generator,
    )
