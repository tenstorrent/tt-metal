# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
import torchvision
import pytest
import evaluate
import os
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from models.utility_functions import (
    torch2tt_tensor,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.efficientnet.tt.efficientnet_model import efficientnet_b0
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES as labels_dict


def make_input_tensor(imagenet_sample_input, resize=256, crop=224):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform(imagenet_sample_input)


def data_loader(iterations, folder_path):
    random.seed(264)
    all_images = os.listdir(folder_path)
    random.shuffle(all_images)
    image_names = all_images[:iterations]

    ground_truth = []
    images = []

    transform = transforms.ToTensor()
    for i in range(iterations):
        key_from_image_name = image_names[i].split("_")[-1].split(".")[0]
        index = list(labels_dict.keys()).index(key_from_image_name)
        ground_truth.append(index)
        image_path = os.path.join(folder_path, image_names[i])
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)

    return images, ground_truth


def run_perf_efficientnet_b0(
    imagenet_sample_input,
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    iterations,
    device,
):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    cpu_key = "ref_key"

    test_input = make_input_tensor(imagenet_sample_input)

    hf_model = torchvision.models.efficientnet_b0()
    tt_input = torch2tt_tensor(test_input, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_model = efficientnet_b0(device)

    folder_path = str(model_location_generator("ImageNet_data"))
    images, ground_truth = data_loader(iterations, folder_path)
    predicted_label = []

    with torch.no_grad():
        profiler.start(cpu_key)
        hf_model(test_input)
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_input)
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_input)
        torch_op = tt_to_torch_tensor(tt_output)
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_output

        profiler.start(third_key)
        for i in range(iterations):
            tt_input = make_input_tensor(images[i])
            tt_input = torch_to_tt_tensor_rm(images[i], device, put_on_device=False)
            tt_output = tt_model(tt_input)
            torch_op_tt = tt_to_torch_tensor(tt_output)
            tt_prediction = torch.argmax(torch_op_tt)
            predicted_label.append(tt_prediction.item())
        ttnn.synchronize_device(device)
        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(references=ground_truth, predictions=predicted_label)
    logger.info(f"Accuracy: {accuracy}")

    prep_perf_report("EfficientNet", 1, first_iter_time, second_iter_time, 100, 100, "b0", cpu_time)

    logger.info(f"efficientnet_b0 inference time: {second_iter_time}")
    logger.info(f"efficientnet_b0 compile time: {compile_time}")
    logger.info(f"efficientnet_b0 inference for {iterations} samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            6.0,
            16.5,
            10,
        ),
    ),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    imagenet_sample_input,
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    iterations,
):
    run_perf_efficientnet_b0(
        imagenet_sample_input,
        model_location_generator,
        expected_inference_time,
        expected_compile_time,
        iterations,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            6.0,
            16.5,
            10,
        ),
    ),
)
def test_perf_virtual_machine(
    device,
    use_program_cache,
    imagenet_sample_input,
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    iterations,
):
    run_perf_efficientnet_b0(
        imagenet_sample_input,
        model_location_generator,
        expected_inference_time,
        expected_compile_time,
        iterations,
        device,
    )
