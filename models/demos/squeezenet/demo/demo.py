# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from PIL import Image
from torchvision import models, transforms
from loguru import logger
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.squeezenet.tt.tt_squeezenet import tt_squeezenet


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = ttnn.from_torch(model.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)

    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def run_demo_dataset(batch_size, device, iterations):
    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    torch_squeezenet.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_squeezenet, custom_preprocessor=custom_preprocessor, device=None
    )
    filename = "models/demos/squeezenet/demo/dog_image.jpeg"
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.repeat(batch_size, 1, 1, 1)
    torch_out = torch_squeezenet(input_batch)
    correct = 0
    for iters in range(iterations):
        tt_input = ttnn.from_torch(torch.permute(input_batch, (0, 2, 3, 1)))
        tt_out = tt_squeezenet(device=device, parameters=parameters, tt_input=tt_input)
        tt_out_torch = ttnn.to_torch(tt_out)
        _, torch_predicted = torch.max(torch_out.data, -1)
        _, ttnn_predicted = torch.max(tt_out_torch.data, -1)
        for i in range(batch_size):
            logger.info(f"Iter: {iters} Sample {i}:")
            logger.info(f"torch Label: {torch_predicted[i]}")
            logger.info(f"Predicted Label: {ttnn_predicted[i]}")

            if torch_predicted[i] == ttnn_predicted[i]:
                correct += 1

    accuracy = correct / (batch_size * iterations)
    logger.info(f"Dataset Inference Accuracy for {batch_size}x{iterations} Samples : {accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("iterations", [1])
def test_demo_dataset(
    device,
    batch_size,
    iterations,
    model_location_generator,
    reset_seeds,
):
    return run_demo_dataset(
        device=device,
        batch_size=batch_size,
        iterations=iterations,
    )
