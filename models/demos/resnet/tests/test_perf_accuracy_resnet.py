# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import AutoImageProcessor
import pytest
import numpy as np
from loguru import logger
import ttnn

from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
    is_e75,
    skip_for_wormhole_b0,
)
from models.perf.perf_utils import prep_perf_report
from models.demos.resnet.tests.demo_utils import get_data, load_resnet50_model
from models.demos.resnet.tt.metalResnetBlock50 import ResNet, Bottleneck
from datasets import load_dataset

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi2,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def run_perf_resnet(
    model_location_generator,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    iterations,
    device,
):
    disable_persistent_kernel_cache()
    first_key = f"first_iter"
    second_key = f"second_iter"
    third_key = f"accuracy_loop"
    cpu_key = f"ref_key"
    model_name = "microsoft/resnet-50"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"].repeat([batch_size, 1, 1, 1])
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    torch_resnet50 = load_resnet50_model(model_location_generator)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    sharded = False
    if batch_size >= 8:
        sharded = True
    tt_resnet50 = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        device=device,
        state_dict=state_dict,
        base_address="",
        fold_batchnorm=True,
        storage_in_dram=False,
        batch_size=batch_size,
        model_config=model_config,
        sharded=sharded,
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        tt_inputs = tt_resnet50.preprocessing(inputs)
        profiler.start(first_key)
        tt_output = tt_resnet50(tt_inputs)
        tt_output = tt_output.cpu().to_torch().to(torch.float)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        tt_inputs = tt_resnet50.preprocessing(inputs)
        profiler.start(second_key)
        tt_output = tt_resnet50(tt_inputs)
        tt_output = tt_output.cpu().to_torch().to(torch.float)
        profiler.end(second_key)
        del tt_output

        logger.info("ImageNet-1k validation Dataset")
        if iterations <= 50:
            input_loc = str(model_location_generator("sample_data"))
        else:
            input_loc = str(model_location_generator("ImageNet_data"))
        image_examples = get_data(input_loc)
        reference_labels = []
        predicted_labels = []
        profiler.start(third_key)

        weka_is_on = True
        if len(image_examples) == 0:
            weka_is_on = False
            files_raw = iter(load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True))
            image_examples = []
            sample_count = batch_size * iterations
            for _ in range(sample_count):
                image_examples.append(next(files_raw))

        for i in range(iterations):
            if weka_is_on:
                input_image = image_examples[i].image
            else:
                input_image = image_examples[i]["image"]
            if input_image.mode == "L":
                input_image = input_image.convert(mode="RGB")
            input = image_processor(input_image, return_tensors="pt")
            input = input["pixel_values"].repeat([batch_size, 1, 1, 1])
            tt_inputs = tt_resnet50.preprocessing(input)
            tt_output = tt_resnet50(tt_inputs)
            tt_output = tt_output.cpu().to_torch().to(torch.float)
            for j in range(batch_size):
                prediction = tt_output[j][0][0].argmax()
                prediction = prediction.item()
                predicted_labels.append(prediction)
                if weka_is_on:
                    reference_labels.append(image_examples[i].label)
                else:
                    reference_labels.append(image_examples[i]["label"])
        predicted_labels = np.array(predicted_labels)
        reference_labels = np.array(reference_labels)
        accuracy = np.mean(predicted_labels == reference_labels)
        logger.info("Accuracy")
        logger.info(accuracy)
        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_perf_report(
        model_name=f"resnet50",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"resnet50 inference time: {second_iter_time}")
    logger.info(f"resnet50 compile time: {compile_time}")
    logger.info(f"resnet50 inference for {batch_size}x{iterations} Samples: {third_iter_time}")


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time, iterations",
    ((16, 0.015, 14.5, 160), (20, 0.014, 14.5, 160)),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    model_location_generator,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    iterations,
    function_level_defaults,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")

    run_perf_resnet(
        model_location_generator,
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        iterations,
        device,
    )
