# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoImageProcessor

import ttnn
from models.demos.mobilenetv2.tests.mobilenetv2_e2e_performant import MobileNetV2Trace2CQ
from models.demos.ttnn_resnet.tests.demo_utils import get_batch, get_data_loader
from models.utility_functions import profiler, run_for_wormhole_b0

NUM_VALIDATION_IMAGES_IMAGENET = 49920


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, iterations, act_dtype, weight_dtype",
    ((8, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_mobilenetv2_imagenet_demo(
    device,
    batch_size_per_device,
    iterations,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator,
    entire_imagenet_dataset=False,
    expected_accuracy=0.68,
):
    batch_size = batch_size_per_device * device.get_num_devices()
    iterations = iterations // device.get_num_devices()

    if entire_imagenet_dataset:
        iterations = NUM_VALIDATION_IMAGES_IMAGENET // batch_size

    profiler.clear()
    with torch.no_grad():
        mobilenetv2_trace_2cq = MobileNetV2Trace2CQ()

        profiler.start(f"compile")
        mobilenetv2_trace_2cq.initialize_mobilenetv2_trace_2cqs_inference(
            device,
            batch_size_per_device,
            act_dtype,
            weight_dtype,
        )
        profiler.end(f"compile")
        model_version = "microsoft/resnet-50"
        image_processor = AutoImageProcessor.from_pretrained(model_version)
        logger.info("ImageNet-1k validation Dataset")
        input_loc = str(model_location_generator("ImageNet_data"))
        data_loader = get_data_loader(input_loc, batch_size, iterations, entire_imagenet_dataset)

        input_tensors_all = []
        input_labels_all = []
        for iter in tqdm(range(iterations), desc="Preparing images"):
            inputs, labels = get_batch(data_loader, image_processor)
            input_tensors_all.append(inputs)
            input_labels_all.append(labels)
        logger.info("Processed ImageNet-1k validation Dataset")

        logger.info("Starting inference")
        correct = 0
        total_inference_time = 0
        for iter in range(iterations):
            predictions = []
            torch_input_tensor = input_tensors_all[iter]
            labels = input_labels_all[iter]
            profiler.start(f"run")

            n, c, h, w = torch_input_tensor.shape
            torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
            torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
            tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
            output = mobilenetv2_trace_2cq.execute_mobilenetv2_trace_2cqs_inference(tt_inputs_host)
            output = ttnn.to_torch(output)
            prediction = output.argmax(dim=-1)

            profiler.end(f"run")
            total_inference_time += profiler.get(f"run")
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
        mobilenetv2_trace_2cq.release_mobilenetv2_trace_2cqs_inference()
        accuracy = correct / (batch_size * iterations)
        logger.info(f"=============")
        logger.info(f"Accuracy for  batch size: {batch_size} over {iterations} iterations is: {accuracy}")
        if entire_imagenet_dataset:
            assert (
                accuracy < expected_accuracy
            ), f"Accuracy {accuracy} does not match expected accuracy {expected_accuracy}"

        first_iter_time = profiler.get(f"compile")
        inference_time_avg = total_inference_time / (iterations)

        compile_time = first_iter_time - 2 * inference_time_avg
