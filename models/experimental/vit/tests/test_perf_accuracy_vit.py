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
from models.experimental.vit.vit_helper_funcs import get_data_loader, get_batch
from loguru import logger

BATCH_SIZE = 1


def run_perf_vit(
    expected_inference_time,
    expected_compile_time,
    iterations,
    hf_cat_image_sample_input,
    model_location_generator,
    imagenet_label_dict,
    device,
):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    third_key = "Vit_for_image_classification"
    batch_size = BATCH_SIZE
    image = hf_cat_image_sample_input

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    inputs = image_processor(image, return_tensors="pt")
    HF_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

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

        input_loc = str(model_location_generator("ImageNet_data"))
        data_loader = get_data_loader(input_loc, batch_size, iterations)
        profiler.start(third_key)
        correct = 0
        for iter in range(iterations):
            predictions = []
            inputs, labels = get_batch(data_loader, image_processor)
            tt_inputs = torch_to_tt_tensor_rm(inputs, device, put_on_device=False)

            tt_inputs = tt_inputs.to(
                device,
                ttnn.L1_MEMORY_CONFIG,
            )
            tt_output = tt_model(tt_inputs)[0]
            tt_output = tt_output.cpu().to_torch().to(torch.float)
            prediction = tt_output[:, 0, 0, :].argmax(dim=-1)
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
            del tt_output, tt_inputs, inputs, labels, predictions
        accuracy = correct / (batch_size * iterations)
        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
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
    logger.info(f"vit inference for {batch_size}x{iterations} Samples: {third_iter_time}")
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")


@pytest.mark.skip(reason="#7527: Test needs review")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time,iterations",
    (
        (
            2.1,
            16,
            50,
        ),
    ),
)
def test_perf_bare_metal(
    expected_inference_time,
    expected_compile_time,
    iterations,
    hf_cat_image_sample_input,
    model_location_generator,
    imagenet_label_dict,
    device,
):
    run_perf_vit(
        expected_inference_time,
        expected_compile_time,
        iterations,
        hf_cat_image_sample_input,
        model_location_generator,
        imagenet_label_dict,
        device,
    )


@pytest.mark.skip(reason="#7527: Test needs review")
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time,iterations",
    (
        (
            2.9,
            17.5,
            50,
        ),
    ),
)
def test_perf_virtual_machine(
    expected_inference_time,
    expected_compile_time,
    iterations,
    hf_cat_image_sample_input,
    model_location_generator,
    imagenet_label_dict,
    device,
):
    run_perf_vit(
        expected_inference_time,
        expected_compile_time,
        iterations,
        hf_cat_image_sample_input,
        model_location_generator,
        imagenet_label_dict,
        device,
    )
