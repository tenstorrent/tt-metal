# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import timm
import ttnn
import torch
import pytest
from loguru import logger

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_grayskull, is_wormhole_b0
from models.experimental.functional_vovnet.tests.demo_utils import *
from models.experimental.functional_vovnet.tt import ttnn_functional_vovnet


def run_demo(
    device,
    reset_seeds,
    input_path,
    model_version,
    batch_size,
    imagenet_label_dict,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    model = timm.create_model(model_version, pretrained=True).eval()

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )
    profiler.start(f"preprocessing_parameter")
    data_loader = get_data_loader(input_path, batch_size, 2)
    torch_pixel_values, labels = get_batch(data_loader, model)
    ttnn_input = ttnn.from_torch(torch_pixel_values.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)

    ttnn_correct = 0
    ttnn_predictions = []
    dataset_predictions = []

    profiler.start(f"inference_time")
    ttnn_output = ttnn_functional_vovnet.vovnet(
        device=device,
        x=ttnn_input,
        torch_model=model.state_dict(),
        parameters=parameters,
        model=model,
        batch_size=batch_size,
        layer_per_block=3,
        residual=False,
        depthwise=True,
        debug=False,
        bias=False,
    )
    profiler.end(f"inference_time")

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.squeeze(0).squeeze(0)

    for i in range(batch_size):
        dataset_predictions.append(imagenet_label_dict[labels[i]])
        ttnn_predictions.append(imagenet_label_dict[ttnn_output[i].argmax(-1).item()])
        logger.info(f"Sample {i} - Expected Label: {dataset_predictions[i]} - Predicted Label: {ttnn_predictions[i]}")

        if dataset_predictions[i] == ttnn_predictions[i]:
            ttnn_correct += 1

    ttnn_accuracy = ttnn_correct / (batch_size)
    logger.info(f"Inference Accuracy : {ttnn_accuracy}")

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "inference_time": profiler.get("inference_time"),
    }

    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")

    return measurements


def run_demo_imagenet_1k(
    device,
    reset_seeds,
    model_version,
    batch_size,
    iterations,
    imagenet_label_dict,
    model_location_generator,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    profiler.clear()

    model = timm.create_model(model_version, pretrained=True).eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    logger.info("ImageNet-1k validation Dataset")
    input_path = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_path, batch_size, iterations)

    correct = 0

    for iter in range(iterations):
        torch_pixel_values, labels = get_batch(data_loader, model)
        ttnn_input = ttnn.from_torch(torch_pixel_values.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)

        ttnn_correct = 0
        dataset_ttnn_correct = 0
        ttnn_predictions = []
        dataset_predictions = []

        ttnn_output = ttnn_functional_vovnet.vovnet(
            device=device,
            x=ttnn_input,
            torch_model=model.state_dict(),
            parameters=parameters,
            model=model,
            batch_size=batch_size,
            layer_per_block=3,
            residual=False,
            depthwise=True,
            debug=False,
            bias=False,
        )

        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.squeeze(0).squeeze(0)

        for i in range(batch_size):
            dataset_predictions.append(imagenet_label_dict[labels[i]])
            ttnn_predictions.append(imagenet_label_dict[ttnn_output[i].argmax(-1).item()])
            # logger.info(f"Iter: {iter} Sample {i}:")
            # logger.info(f"Expected Label: {dataset_predictions[i]}")
            # logger.info(f"Predicted Label: {ttnn_predictions[i]}")

            if dataset_predictions[i] == ttnn_predictions[i]:
                dataset_ttnn_correct += 1
                correct += 1
        dataset_ttnn_accuracy = dataset_ttnn_correct / (batch_size)
        logger.info(
            f"ImageNet Inference Accuracy for iter {iter} of {batch_size} input samples : {dataset_ttnn_accuracy}"
        )

    accuracy = correct / (batch_size * iterations)
    logger.info(f"ImageNet Inference Accuracy for {batch_size}x{iterations} Samples : {accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
# @pytest.mark.parametrize("batch_size", [7 if is_wormhole_b0() else 1])
@pytest.mark.parametrize(
    "model_version",
    ["hf_hub:timm/ese_vovnet19b_dw.ra_in1k"],
)
def test_demo(
    device,
    reset_seeds,
    input_path,
    model_version,
    batch_size,
    imagenet_label_dict,
):
    return run_demo(
        device=device,
        reset_seeds=reset_seeds,
        input_path=input_path,
        model_version=model_version,
        batch_size=batch_size,
        imagenet_label_dict=imagenet_label_dict,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
# @pytest.mark.parametrize("batch_size", [7 if is_wormhole_b0() else 1])
@pytest.mark.parametrize("iterations", [3])
@pytest.mark.parametrize(
    "model_version",
    ["hf_hub:timm/ese_vovnet19b_dw.ra_in1k"],
)
def test_demo_imagenet_1k(
    device,
    reset_seeds,
    model_version,
    batch_size,
    iterations,
    imagenet_label_dict,
    model_location_generator,
):
    return run_demo_imagenet_1k(
        device=device,
        reset_seeds=reset_seeds,
        model_version=model_version,
        batch_size=batch_size,
        iterations=iterations,
        imagenet_label_dict=imagenet_label_dict,
        model_location_generator=model_location_generator,
    )
