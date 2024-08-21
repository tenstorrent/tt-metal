# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest

from PIL import Image
from loguru import logger
from collections import defaultdict

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.utility_functions import Profiler
from models.perf.perf_utils import prep_perf_report
from models.experimental.ssd.tt.ssd_lite import ssd_for_object_detection
from models.experimental.ssd.reference.utils.metrics import load_ground_truth_labels, calculate_ap

import torchvision.transforms as transforms
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)


BATCH_SIZE = 1


def run_perf_ssd(
    device,
    expected_inference_time,
    expected_compile_time,
    imagenet_sample_input,
    model_location_generator,
    iterations,
):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_key"
    cpu_key = "ref_key"
    comments = "SSD_lite"

    torch_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    torch_model.eval()

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=True)

    tt_model = ssd_for_object_detection(device)
    tt_model.eval()

    data_path = model_location_generator("coco128", model_subdir="ssd")
    image_files = os.listdir(data_path / "images/train2017")
    label_files = os.listdir(data_path / "labels/train2017")

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = torch_model(imagenet_sample_input)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_input)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_input)
        profiler.end(second_key)

        all_predictions = []
        all_ground_truths = []
        iteration = 0

        profiler.start(third_key)
        while iteration < iterations:
            image_file = image_files[iteration]
            image = os.path.join(data_path / "images/train2017", image_file)

            label_file = image_file.split(".")[0] + ".txt"
            label = os.path.join(data_path / "labels/train2017", label_file)

            if os.path.exists(label):
                reference_labels = load_ground_truth_labels(label)
                all_ground_truths.append(reference_labels)

                image = Image.open(image)
                image = image.resize((224, 224))
                image = transforms.ToTensor()(image).unsqueeze(0)

                tt_model = ssd_for_object_detection(device)
                tt_model.eval()
                tt_input = torch_to_tt_tensor_rm(image, device)
                tt_output = tt_model(tt_input)

                class_confidence = defaultdict(float)
                class_bbox = {}

                for i in range(len(tt_output[0]["scores"])):
                    label = int(tt_output[0]["labels"][i])
                    confidence = float(tt_output[0]["scores"][i])
                    bbox = [float(coord) for coord in tt_output[0]["boxes"][i]]

                    if confidence > class_confidence[label]:
                        class_confidence[label] = confidence
                        class_bbox[label] = bbox

                for label, confidence in class_confidence.items():
                    prediction = {"class": label, "confidence": confidence, "bbox": class_bbox[label]}
                    all_predictions.append(prediction)

            iteration += 1

        ap = calculate_ap(all_predictions, all_ground_truths)
        logger.info(f"Mean Average Precision (mAP): {ap}")

        profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)

    prep_perf_report(
        model_name="SSD",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_inference_time,
        expected_inference_time=expected_compile_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time

    logger.info(f"SSD inference time: {second_iter_time}")
    logger.info(f"SSD compile time: {compile_time}")
    logger.info(f"ssd inference for {iterations} samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            13.48890358,
            0.5188875198,
            50,
        ),
    ),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    imagenet_sample_input,
    model_location_generator,
    iterations,
):
    run_perf_ssd(
        device,
        expected_inference_time,
        expected_compile_time,
        imagenet_sample_input,
        model_location_generator,
        iterations,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations,",
    (
        (
            13.48890358,
            0.5188875198,
            50,
        ),
    ),
)
def test_perf_virtual_machine(
    device,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    imagenet_sample_input,
    model_location_generator,
    iterations,
):
    run_perf_ssd(
        device,
        expected_inference_time,
        expected_compile_time,
        imagenet_sample_input,
        model_location_generator,
        iterations,
    )
