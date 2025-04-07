# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger

from models.utility_functions import run_for_wormhole_b0
from models.demos.ttnn_resnet.tests.demo_utils import get_data_loader, get_batch
from transformers import AutoImageProcessor
from models.utility_functions import (
    profiler,
)
from models.experimental.functional_mobilenetv2.test.mobilenetv2_performant_imagenet import MobileNetV2Trace2CQ


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_mobilenetv2_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    enable_async_mode,
    model_location_generator,
):
    profiler.clear()
    with torch.no_grad():
        # For MobileNetV2 model initialization
        mobilenetv2_trace_2cq = MobileNetV2Trace2CQ()

        profiler.start(f"compile")
        mobilenetv2_trace_2cq.initialize_mobilenetv2_trace_2cqs_inference(
            device,
            batch_size,
            act_dtype,
            weight_dtype,
        )
        profiler.end(f"compile")

        model_version = "google/mobilenet_v2_1.0_224"
        iterations = 100
        image_processor = AutoImageProcessor.from_pretrained(model_version)
        input_loc = str(model_location_generator("ImageNet_data"))
        data_loader = get_data_loader(input_loc, batch_size, iterations)

        # input_tensors_all = []
        # input_labels_all = []
        # for iter in range(iterations):
        # input_tensors_all.append(inputs)
        # input_labels_all.append(labels)

        correct = 0
        profiler.start(f"run")
        for iter in range(iterations):
            predictions = []
            inputs, labels = get_batch(data_loader, image_processor)
            # torch_input_tensor = inputs.reshape(batch_size, 3, 224, 224)
            torch_input_tensor = torch.permute(inputs, (0, 2, 3, 1))

            ttnn_input_tensor = torch_input_tensor.reshape(
                1,
                1,
                torch_input_tensor.shape[0] * torch_input_tensor.shape[1] * torch_input_tensor.shape[2],
                torch_input_tensor.shape[3],
            )

            ### TODO optimize input streamer for better e2e performance
            tt_inputs_host = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            output = mobilenetv2_trace_2cq.execute_mobilenetv2_trace_2cqs_inference(tt_inputs_host)
            output = ttnn.from_device(output, blocking=True).to_torch().to(torch.float)
            prediction = output.argmax(dim=-1)
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
            del output, inputs, labels, predictions
        profiler.end(f"run")
        mobilenetv2_trace_2cq.release_mobilenetv2_trace_2cqs_inference()
        accuracy = correct / (batch_size * iterations)
        logger.info(f"=============")
        logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")

        first_iter_time = profiler.get(f"compile")
        # ensuring inference time fluctuations is not noise
        inference_time_avg = profiler.get("run") / (iterations)

        compile_time = first_iter_time - 2 * inference_time_avg
    logger.info(
        f"ttnn_{model_version}_batch_size{batch_size} tests inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"ttnn_{model_version}_batch_size{batch_size} compile time: {compile_time}")
