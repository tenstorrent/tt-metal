# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import transformers
from loguru import logger
from tqdm import tqdm
from transformers import AutoImageProcessor

import ttnn
from models.demos.vit.tests.vit_performant_imagenet import VitTrace2CQ
from models.demos.wormhole.vit.demo.vit_helper_funcs import get_batch, get_data_loader
from models.utility_functions import profiler, run_for_wormhole_b0

NUM_VALIDATION_IMAGES_IMAGENET = 49920


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1700000}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, iterations",
    ((8, 100),),
)
def test_run_vit_trace_2cqs_inference(
    mesh_device,
    batch_size_per_device,
    iterations,
    imagenet_label_dict,
    model_location_generator,
    entire_imagenet_dataset=False,
    expected_accuracy=0.80,
):
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    iterations = iterations // mesh_device.get_num_devices()

    if entire_imagenet_dataset:
        iterations = NUM_VALIDATION_IMAGES_IMAGENET // batch_size

    profiler.clear()
    with torch.no_grad():
        vit_trace_2cq = VitTrace2CQ()

        profiler.start(f"compile")
        vit_trace_2cq.initialize_vit_trace_2cqs_inference(
            mesh_device,
            batch_size_per_device,
        )
        profiler.end(f"compile")

        model_version = "google/vit-base-patch16-224"
        image_processor = AutoImageProcessor.from_pretrained(model_version)
        config = transformers.ViTConfig.from_pretrained(model_version)

        logger.info("ImageNet-1k validation Dataset")
        input_loc = str(model_location_generator("ImageNet_data"))
        data_loader = get_data_loader(input_loc, batch_size, iterations, entire_imagenet_dataset)

        input_tensors_all = []
        input_labels_all = []
        for iter in tqdm(range(iterations), desc="Preparing images"):
            inputs, labels = get_batch(data_loader, image_processor)
            # preprocessing
            inputs = torch.permute(inputs, (0, 2, 3, 1))
            inputs = torch.nn.functional.pad(inputs, (0, 1, 0, 0, 0, 0, 0, 0))
            batch_size, img_h, img_w, img_c = inputs.shape  # permuted input NHWC
            patch_size = config.patch_size
            inputs = inputs.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
            #
            input_tensors_all.append(inputs)
            input_labels_all.append(labels)
        logger.info("Processed ImageNet-1k validation Dataset")

        logger.info("Starting inference")
        correct = 0
        total_inference_time = 0
        for iter in range(iterations):
            predictions = []
            inputs = input_tensors_all[iter]
            labels = input_labels_all[iter]
            profiler.start(f"run")
            tt_inputs_host = ttnn.from_torch(
                inputs,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=vit_trace_2cq.test_infra.inputs_mesh_mapper,
            )
            output = vit_trace_2cq.execute_vit_trace_2cqs_inference(tt_inputs_host)
            output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            # 1000 classes trimming
            prediction = output[:, 0, :1000].argmax(dim=-1)
            profiler.end(f"run")
            total_inference_time += profiler.get(f"run")
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
        vit_trace_2cq.release_vit_trace_2cqs_inference()
        accuracy = correct / (batch_size * iterations)
        logger.info(f"=============")
        logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
        if entire_imagenet_dataset:
            assert (
                accuracy == expected_accuracy
            ), f"Accuracy {accuracy} does not match expected accuracy {expected_accuracy}"

        first_iter_time = profiler.get(f"compile")
        # ensuring inference time fluctuations is not noise
        inference_time_avg = total_inference_time / (iterations)

        compile_time = first_iter_time - 2 * inference_time_avg
    logger.info(
        f"ttnn_{model_version}_batch_size{batch_size} tests inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"ttnn_{model_version}_batch_size{batch_size} compile time: {compile_time}")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1700000, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, iterations",
    ((8, 100),),
)
@pytest.mark.parametrize("entire_imagenet_dataset", [True])
@pytest.mark.parametrize("expected_accuracy", [0.80])
def test_run_vit_trace_2cqs_accuracy(
    mesh_device,
    batch_size_per_device,
    iterations,
    imagenet_label_dict,
    model_location_generator,
    entire_imagenet_dataset,
    expected_accuracy,
):
    test_run_vit_trace_2cqs_inference(
        mesh_device,
        batch_size_per_device,
        iterations,
        imagenet_label_dict,
        model_location_generator,
        entire_imagenet_dataset,
        expected_accuracy,
    )
