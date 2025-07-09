# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.swin_s.demo.demo_utils import get_batch, get_data_loader
from models.experimental.swin_s.runner.performant_runner import SwinSPerformantRunner
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0

imagenet_label_dict = {i: label for i, label in enumerate(IMAGENET2012_CLASSES)}
from loguru import logger


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 16998400, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
def test_run_swin_s_trace_2cqs_inference(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    disable_persistent_kernel_cache()

    swin_s_trace_2cq = SwinSPerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=None,
    )

    batch_size = 1
    iterations = 100

    logger.info("ImageNet-1k validation Dataset")
    input_loc = "models/demos/swin_s/demo/ImageNet_data"
    data_loader = get_data_loader(input_loc, batch_size, iterations)
    inputs, labels = get_batch(data_loader)
    output_tensor = swin_s_trace_2cq.run(inputs)

    output_tensor = ttnn.to_torch(output_tensor).to(torch.float)

    prediction = output_tensor.argmax(dim=-1)
    predictions = []
    correct = 0
    for i in range(batch_size):
        predictions.append(imagenet_label_dict[prediction[i].item()])
        logger.info(
            f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
        )
        if imagenet_label_dict[labels[i]] == predictions[-1]:
            correct += 1

        del output_tensor, inputs, labels, predictions

    swin_s_trace_2cq.release()
