# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_data_loader, get_batch, load_imagenet_dataset, get_mesh_mappers
from models.experimental.efficientnetb0.runner.performant_runner import EfficientNetb0PerformantRunner
from models.experimental.efficientnetb0.common import load_torch_model, EFFICIENTNETB0_L1_SMALL_SIZE


def run_demo(
    model_type,
    source,
    device,
    reset_seeds,
    batch_size_per_device,
    imagenet_label_dict,
    model_location_generator=None,
    resolution=224,
):
    logger.info("ImageNet-1k validation Dataset")
    input_loc = load_imagenet_dataset(model_location_generator)
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    data_loader = get_data_loader(input_loc, batch_size, 1)
    input_tensor, labels = get_batch(data_loader, resolution=resolution)

    if model_type == "torch_model":
        torch_model = load_torch_model(model_location_generator)
        output = torch_model(input_tensor)
        logger.info("Inferencing [Torch] Model")
    else:
        logger.info(f"Running with batch_size = {batch_size} across {num_devices} devices")
        inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)
        performant_runner = EfficientNetb0PerformantRunner(
            device,
            batch_size_per_device,
            ttnn.bfloat16,
            ttnn.bfloat16,
            model_location_generator=model_location_generator,
            resolution=(resolution, resolution),
            mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            mesh_composer=outputs_mesh_composer,
        )
        output = performant_runner.run(torch_input_tensor=input_tensor)
        output = ttnn.to_torch(output, mesh_composer=outputs_mesh_composer)
        performant_runner.release()
        logger.info("Inferencing [TTNN] Model")
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, predicted_id = torch.topk(probabilities, 3)

    correct = 0
    predictions = []
    for i in range(batch_size_per_device):
        predictions.append(imagenet_label_dict[predicted_id[i].item()])
        logger.info(
            f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
        )
        if imagenet_label_dict[labels[i]] == predictions[-1]:
            correct += 1


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": EFFICIENTNETB0_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "source",
    [
        "models/experimental/efficientnetb0/demo/input_image.jpg",
    ],
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
def test_demo(
    model_type, source, device, reset_seeds, batch_size_per_device, imagenet_label_dict, model_location_generator
):
    run_demo(
        model_type, source, device, reset_seeds, batch_size_per_device, imagenet_label_dict, model_location_generator
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": EFFICIENTNETB0_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "source",
    [
        "models/experimental/efficientnetb0/demo/input_image.jpg",
    ],
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
def test_demo_dp(
    model_type, source, mesh_device, reset_seeds, batch_size_per_device, imagenet_label_dict, model_location_generator
):
    run_demo(
        model_type,
        source,
        mesh_device,
        reset_seeds,
        batch_size_per_device,
        imagenet_label_dict,
        model_location_generator,
    )
