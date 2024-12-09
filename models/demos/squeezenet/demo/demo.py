# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from torchvision import models
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
from models.demos.squeezenet.demo_utils import get_data_loader, get_batch, preprocess
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


def run_squeezenet_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, device):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    profiler.clear()

    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    torch_squeezenet.to(torch.bfloat16)
    torch_squeezenet.eval()
    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_squeezenet, custom_preprocessor=custom_preprocessor, device=None
    )
    profiler.end(f"preprocessing_parameter")
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)
    correct = 0
    torch_ttnn_correct = 0
    torch_correct = 0
    for iter in range(iterations):
        predictions = []
        torch_predictions = []
        inputs, labels = get_batch(data_loader)
        torch_outputs = torch_squeezenet(inputs)
        profiler.start(f"preprocessing_input")
        tt_batched_input_tensor = ttnn.from_torch(
            inputs.permute(0, 2, 3, 1),
            ttnn.bfloat16,
        )
        profiler.end(f"preprocessing_input")
        profiler.start(f"inference_time")
        tt_output = tt_squeezenet(device=device, parameters=parameters, tt_input=tt_batched_input_tensor)
        profiler.end(f"inference_time")
        tt_output = ttnn.to_torch(tt_output)
        prediction = tt_output.argmax(dim=-1)
        torch_prediction = torch_outputs.argmax(dim=-1)
        for i in range(batch_size):
            if prediction.dim() == 0:
                predictions.append(imagenet_label_dict[prediction.item()])
            else:
                predictions.append(imagenet_label_dict[prediction[i].item()])

            if torch_prediction.dim() == 0:
                torch_predictions.append(imagenet_label_dict[torch_prediction.item()])
            else:
                torch_predictions.append(imagenet_label_dict[torch_prediction[i].item()])

            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- \n Torch Predicted label:{predictions[-1]} \tPredicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1
            if imagenet_label_dict[labels[i]] == torch_predictions[-1]:
                torch_correct += 1
            if predictions[-1] == torch_predictions[-1]:
                torch_ttnn_correct += 1

        del tt_output, tt_batched_input_tensor, inputs, labels, predictions
    accuracy = correct / (batch_size * iterations)
    torch_accuracy = torch_correct / (batch_size * iterations)
    torch_ttnn_accuracy = torch_ttnn_correct / (batch_size * iterations)

    logger.info(f"Model SqueezeNet for Image Classification")
    logger.info(f"TTNN Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
    logger.info(f"Torch Accuracy for {batch_size}x{iterations} inputs: {torch_accuracy}")
    logger.info(f"Torch vs TTNN Accuracy for {batch_size}x{iterations} inputs: {torch_ttnn_accuracy}")
    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
    }
    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    assert (
        torch_ttnn_accuracy >= 0.90
    ), f"Torch vs TTNN Accuracy : {0.90} Expected Torch vs TTNN Accuracy accuracy: {torch_ttnn_accuracy}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((8, 10),),
)
def test_demo_dataset(batch_size, iterations, imagenet_label_dict, model_location_generator, device):
    return run_squeezenet_imagenet_inference(
        batch_size, iterations, imagenet_label_dict, model_location_generator, device
    )
