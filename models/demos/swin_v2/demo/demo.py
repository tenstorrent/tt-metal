# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.swin_v2.demo.demo_utils import get_batch, get_data_loader
from models.experimental.swin_v2.runner.performant_runner import SwinV2PerformantRunner
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0

imagenet_label_dict = {i: label for i, label in enumerate(IMAGENET2012_CLASSES)}
from loguru import logger


def run_swin_v2_trace_2cqs_inference(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    iterations=100,
    input_loc="models/experimental/swin_v2/demo/ImageNet_data",
):
    disable_persistent_kernel_cache()
    total_batch_size = batch_size * device.get_num_devices()
    swin_v2_trace_2cq = SwinV2PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )

    batch_size = 1
    logger.info("ImageNet-1k validation Dataset")
    data_loader = get_data_loader(input_loc, total_batch_size, iterations)
    inputs, labels = get_batch(data_loader)
    output_tensor = swin_v2_trace_2cq.run(inputs)
    output_tensor = ttnn.to_torch(output_tensor, mesh_composer=swin_v2_trace_2cq.runner_infra.output_composer).to(
        torch.float
    )
    prediction = output_tensor.argmax(dim=-1)
    predictions = []
    correct = 0
    for i in range(total_batch_size):
        predictions.append(imagenet_label_dict[prediction[i].item()])
        logger.info(
            f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
        )
        if imagenet_label_dict[labels[i]] == predictions[-1]:
            correct += 1

    del output_tensor, inputs, labels, predictions

    swin_v2_trace_2cq.release()


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
def test_swin_v2_demo(device, batch_size, act_dtype, weight_dtype, model_location_generator, resolution):
    return run_swin_v2_trace_2cqs_inference(
        device, batch_size, act_dtype, weight_dtype, model_location_generator, resolution
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 16998400, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
def test_swin_v2_demo_dp(mesh_device, device_batch_size, act_dtype, weight_dtype, model_location_generator, resolution):
    return run_swin_v2_trace_2cqs_inference(
        mesh_device, device_batch_size, act_dtype, weight_dtype, model_location_generator, resolution
    )
