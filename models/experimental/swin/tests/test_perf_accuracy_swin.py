# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import evaluate

import ttnn
import os
import random
from pathlib import Path
from loguru import logger
from transformers import AutoFeatureExtractor
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoImageProcessor

from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES as labels_dict
from models.experimental.swin.tt.swin_for_image_classification import (
    TtSwinForImageClassification,
)
from transformers import SwinForImageClassification as HF_SwinForImageClassification

from models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.swin.swin_utils import get_data


def run_swin_perf(device, model_name, iterations, model_location_generator):
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    cpu_key = "ref_key"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = HF_SwinForImageClassification.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    folder_path = str(model_location_generator("ImageNet_data"))
    image_examples = get_data(folder_path)
    ground_truth = []
    predicted_label = []

    disable_persistent_kernel_cache()
    base_address = f"swin."
    with torch.no_grad():
        torch_model = model

        tt_model = TtSwinForImageClassification(
            config=model.config,
            state_dict=model.state_dict(),
            base_address=base_address,
            device=device,
        )

        transform = transforms.Compose([transforms.ToTensor()])
        profiler.start(cpu_key)
        torch_input = transform(image_examples[0].image)
        torch_input = torch.unsqueeze(torch_input, 0)
        torch_output = torch_model(torch_input)
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        input_image = image_examples[0].image
        input = image_processor(input_image, return_tensors="pt")
        tt_pixel_values = torch_to_tt_tensor_rm(input.pixel_values, device)
        tt_output = tt_model(tt_pixel_values)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()
        profiler.start(second_key)
        tt_pixel_values = torch_to_tt_tensor_rm(input.pixel_values, device)
        tt_output = tt_model(tt_pixel_values)
        profiler.end(second_key)
        del tt_output

        profiler.start(third_key)
        ttnn.synchronize_device(device)
        for i in range(iterations):
            input_image = image_examples[i].image
            input = image_processor(input_image, return_tensors="pt")

            tt_pixel_values = input.pixel_values
            tt_pixel_values = torch_to_tt_tensor_rm(tt_pixel_values, device)
            tt_output = tt_model(tt_pixel_values)
            tt_output_torch = tt_to_torch_tensor(tt_output.logits)
            tt_prediction = torch.argmax(tt_output_torch)

            ground_truth.append(image_examples[i].label)
            predicted_label.append(tt_prediction.item())
            del tt_output, tt_output_torch, tt_prediction
        profiler.end(third_key)

        accuracy_metric = evaluate.load("accuracy")
        accuracy = accuracy_metric.compute(references=ground_truth, predictions=predicted_label)
        logger.info(f"Accuracy: {accuracy}")

        first_iter_time = profiler.get(first_key)
        second_iter_time = profiler.get(second_key)
        third_iter_time = profiler.get(third_key)
        cpu_time = profiler.get(cpu_key)
        compile_time = first_iter_time - second_iter_time

    prep_perf_report("Swin", 1, first_iter_time, second_iter_time, 100, 100, "", cpu_time)
    logger.info(f"Swin inference time: {second_iter_time}")
    logger.info(f"Swin compile time: {compile_time}")
    logger.info(f"Swin inference for {iterations} samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "model_name,iterations",
    (("microsoft/swin-tiny-patch4-window7-224", 20),),
)
def test_perf_bare_metal(device, use_program_cache, model_name, iterations, model_location_generator):
    run_swin_perf(device, model_name, iterations, model_location_generator)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "model_name,iterations",
    (("microsoft/swin-tiny-patch4-window7-224", 20),),
)
def test_perf_virtual_machine(device, use_program_cache, model_name, iterations, model_location_generator):
    run_swin_perf(device, model_name, iterations, model_location_generator)
