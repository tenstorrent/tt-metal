# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
import torch
from loguru import logger

from models.utility_functions import run_for_wormhole_b0
from models.demos.ttnn_resnet.tests.resnet50_performant_imagenet import ResNet50Trace2CQ
from models.demos.ttnn_resnet.tests.demo_utils import get_data, get_data_loader, get_batch
from transformers import AutoImageProcessor
from models.utility_functions import (
    profiler,
)
from models.perf.perf_utils import prep_perf_report
from pdb import set_trace as bp


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_resnet50_trace_2cqs_inference(
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
        resnet50_trace_2cq = ResNet50Trace2CQ()

        resnet50_trace_2cq.initialize_resnet50_trace_2cqs_inference(
            device,
            batch_size,
            act_dtype,
            weight_dtype,
        )
        model_version = "microsoft/resnet-50"
        iterations = 100
        image_processor = AutoImageProcessor.from_pretrained(model_version)
        input_loc = str(model_location_generator("ImageNet_data"))
        data_loader = get_data_loader(input_loc, batch_size, iterations)
        correct = 0

        profiler.start(f"run")
        for iter in range(iterations):
            predictions = []
            inputs, labels = get_batch(data_loader, image_processor)
            tt_inputs_host = ttnn.from_torch(inputs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            output = resnet50_trace_2cq.execute_resnet50_trace_2cqs_inference(tt_inputs_host).to_torch().to(torch.float)
            prediction = output[:, 0, 0, :].argmax(dim=-1)
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
            accuracy = correct / (batch_size * iterations)
            logger.info(f"=============")
            logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
        profiler.end(f"run")

        resnet50_trace_2cq.release_resnet50_trace_2cqs_inference()

    first_iter_time = profiler.get(f"compile") + profiler.get(f"cache")

    # ensuring inference time fluctuations is not noise
    print(profiler.get("run"))
    inference_time_avg = profiler.get("run") / (iterations * batch_size)

    # cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - 2 * inference_time_avg
    prep_perf_report(
        model_name=f"ttnn_{model_version}_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=30,
        expected_inference_time=0.004,
        comments="tests",
    )

    logger.info(
        f"ttnn_{model_version}_batch_size{batch_size} tests inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"ttnn_{model_version}_batch_size{batch_size} compile time: {compile_time}")
