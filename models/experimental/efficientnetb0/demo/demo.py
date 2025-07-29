# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path

from models.experimental.efficientnetb0.reference import efficientnetb0
from efficientnet_pytorch import EfficientNet
from models.utility_functions import run_for_wormhole_b0
from models.experimental.efficientnetb0.demo.demo_utils import get_data_loader, get_batch
from models.experimental.efficientnetb0.runner.performant_runner import EfficientNetb0PerformantRunner
from models.experimental.efficientnetb0.tt.model_preprocessing import get_mesh_mappers

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

import torch
import ttnn
from loguru import logger


def run_demo(
    model_type, source, device, reset_seeds, batch_size_per_device, imagenet_label_dict, model_location_generator=None
):
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size_per_device * (device.get_num_devices()), 1)
    input_tensor, labels = get_batch(data_loader)

    if model_type == "torch_model":
        model = EfficientNet.from_pretrained("efficientnet-b0").eval()
        state_dict = model.state_dict()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        torch_model = efficientnetb0.Efficientnetb0()

        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()

        output = torch_model(input_tensor)
        logger.info("Inferencing [Torch] Model")
    else:
        num_devices = device.get_num_devices()
        batch_size = batch_size_per_device * num_devices
        logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")
        inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)
        performant_runner = EfficientNetb0PerformantRunner(
            device,
            batch_size_per_device,
            ttnn.bfloat16,
            ttnn.bfloat16,
            resolution=(224, 224),
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
    [{"l1_small_size": 7 * 1024, "trace_region_size": 23887872, "num_command_queues": 2}],
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
    [{"l1_small_size": 7 * 1024, "trace_region_size": 23887872, "num_command_queues": 2}],
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
