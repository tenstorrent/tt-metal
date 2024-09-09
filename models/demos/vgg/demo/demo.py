# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import tt_lib
import torch.nn as nn

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)
import ttnn

from models.demos.vgg.demo_utils import get_data, get_data_loader, get_batch, preprocess
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.vgg.tt import ttnn_vgg

vgg_model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


def run_vgg_imagenet_inference_vgg(
    batch_size,
    iterations,
    imagenet_label_dict,
    model_location_generator,
    model_class,
    weights,
    device,
    model_config=vgg_model_config,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    profiler.clear()

    # Setup model
    torch_model = model_class(weights=weights)
    torch_model.to(torch.bfloat16)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_vgg.custom_preprocessor,
    )

    if model_class == models.vgg11:
        ttnn_model = ttnn_vgg.ttnn_vgg11
        model_name = "VGG11"
    else:
        ttnn_model = ttnn_vgg.ttnn_vgg16
        model_name = "VGG16"

    # load inputs
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    # load ImageNet batch by batch
    # and run inference
    correct = 0
    for iter in range(iterations):
        predictions = []
        torch_predictions = []
        inputs, labels = get_batch(data_loader)
        torch_outputs = torch_model(inputs)
        permuted_inputs = torch.permute(inputs, (0, 2, 3, 1))
        tt_batched_input_tensor = ttnn.from_torch(permuted_inputs, ttnn.bfloat16)
        tt_output = ttnn_model(device, tt_batched_input_tensor, parameters, batch_size, model_config)
        tt_output = ttnn.to_torch(tt_output)
        prediction = tt_output[:, 0, 0, :].argmax(dim=-1)
        torch_prediction = torch_outputs[:, :].argmax(dim=-1)
        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            torch_predictions.append(imagenet_label_dict[torch_prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- \n Torch Predicted label:{predictions[-1]} \tPredicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1

        del tt_output, tt_batched_input_tensor, inputs, labels, predictions
    accuracy = correct / (batch_size * iterations)
    logger.info(f"Model {model_name}")
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "model_class, weights",
    [
        (models.vgg11, models.VGG11_Weights.IMAGENET1K_V1),
        (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    ],
)
def test_demo_imagenet_vgg(
    batch_size, iterations, imagenet_label_dict, model_location_generator, model_class, weights, device
):
    run_vgg_imagenet_inference_vgg(
        batch_size, iterations, imagenet_label_dict, model_location_generator, model_class, weights, device
    )
