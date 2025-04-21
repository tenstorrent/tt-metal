# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger
import os
import glob
from PIL import Image
from models.utility_functions import run_for_wormhole_b0
from models.demos.ttnn_resnet.tests.resnet50_performant_imagenet import ResNet50Trace2CQ
from models.demos.ttnn_resnet.tests.demo_utils import get_batch
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES
from datasets import load_dataset
from transformers import AutoImageProcessor
from tqdm import tqdm
from models.utility_functions import (
    profiler,
)


def get_input(image_path):
    img = Image.open(image_path)
    return img


def get_label(image_path):
    _, image_name = image_path.rsplit("/", 1)
    image_name_exact, _ = image_name.rsplit(".", 1)
    _, label_id = image_name_exact.rsplit("_", 1)
    label = list(IMAGENET2012_CLASSES).index(label_id)
    return label


class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def get_data_loader(input_loc, batch_size, iterations):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = glob.glob(data_path)

    def loader():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=get_input(f1),
                    label=get_label(f1),
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    def loader_hf():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=f1["image"],
                    label=f1["label"],
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    if len(files) == 0:
        ds = load_dataset("imagenet-1k", split="validation", use_auth_token=True)
        files_raw = iter(ds)
        files = []

        for item in tqdm(files_raw, total=len(ds), desc="Loading samples"):
            files.append(item)

        del files_raw
        return loader_hf()

    return loader()


def get_batch(loaded_images, image_processor):
    images = None
    labels = []
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode == "L":
            img = img.convert(mode="RGB")
        img = image_processor(img, return_tensors="pt")
        img = img["pixel_values"]

        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)
    return images, labels


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("enable_async_mode", (True,), indirect=True)
def test_run_resnet50_trace_2cqs_inference(
    mesh_device,
    use_program_cache,
    batch_size_per_device,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    enable_async_mode,
    model_location_generator,
):
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    profiler.clear()
    with torch.no_grad():
        resnet50_trace_2cq = ResNet50Trace2CQ()

        resnet50_trace_2cq.initialize_resnet50_trace_2cqs_inference(
            mesh_device,
            batch_size_per_device,
            act_dtype,
            weight_dtype,
        )
        model_version = "microsoft/resnet-50"
        image_processor = AutoImageProcessor.from_pretrained(model_version)
        logger.info("ImageNet-1k validation Dataset")
        input_loc = str(model_location_generator("ImageNet_data"))
        data_loader = get_data_loader(input_loc, batch_size, 0)

        input_tensors_all = []
        input_labels_all = []
        for batch in tqdm(data_loader, desc="Loading images"):
            inputs, labels = get_batch(batch, image_processor)
            input_tensors_all.append(inputs)
            input_labels_all.append(labels)
        logger.info("Processed ImageNet-1k validation Dataset")

        logger.info("Starting inference")
        correct = 0
        total_inference_time = 0
        iteration = 0
        for inputs, labels in zip(input_tensors_all, input_labels_all):
            predictions = []
            profiler.start(f"run")
            tt_inputs_host = ttnn.from_torch(
                inputs,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=resnet50_trace_2cq.test_infra.inputs_mesh_mapper,
            )
            output = resnet50_trace_2cq.execute_resnet50_trace_2cqs_inference(tt_inputs_host)
            output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            prediction = output[:, 0, 0, :].argmax(dim=-1)
            profiler.end(f"run")
            total_inference_time += profiler.get(f"run")
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iteration} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
            iteration += 1
        resnet50_trace_2cq.release_resnet50_trace_2cqs_inference()
        accuracy = correct / (batch_size * iteration)
        logger.info(f"=============")
        logger.info(f"Accuracy: {accuracy}")

        # ensuring inference time fluctuations is not noise
        inference_time_avg = total_inference_time / (iteration)

    print(f"Batch size: {batch_size}, iterations: {iteration}")
    logger.info(
        f"ttnn_{model_version}_batch_size{batch_size} tests inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
