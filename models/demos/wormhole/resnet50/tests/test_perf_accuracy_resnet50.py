# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger
from models.utility_functions import run_for_wormhole_b0
from models.demos.ttnn_resnet.tests.resnet50_performant_imagenet import ResNet50Trace2CQ
from models.demos.ttnn_resnet.tests.demo_utils import get_batch, get_data_loader
from transformers import AutoImageProcessor
from tqdm import tqdm
from models.utility_functions import (
    profiler,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("enable_async_mode", (True,), indirect=True)
def test_run_resnet50_trace_2cqs_inference_accuracy(
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
        data_loader = get_data_loader(input_loc, batch_size, 0, download_entire_dataset=True)

        input_tensors_all = []
        input_labels_all = []
        for _ in tqdm(iter(int, 1), desc="Preprocess images"):
            try:
                inputs, labels = get_batch(data_loader, image_processor)
                input_tensors_all.append(inputs)
                input_labels_all.append(labels)
            except StopIteration:
                logger.info("Processed ImageNet-1k validation Dataset")
                break

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
