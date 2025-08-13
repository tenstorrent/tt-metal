# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import run_for_wormhole_b0
from models.experimental.efficientnetb0.demo.demo_utils import get_data_loader, get_batch
from models.experimental.efficientnetb0.runner.performant_runner import EfficientNetb0PerformantRunner
from models.experimental.efficientnetb0.common import load_torch_model, EFFICIENTNETB0_L1_SMALL_SIZE


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": EFFICIENTNETB0_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
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
def test_demo(model_type, batch_size, device, reset_seeds, imagenet_label_dict, model_location_generator):
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, 1)
    input_tensor, labels = get_batch(data_loader)

    if model_type == "torch_model":
        torch_model = load_torch_model(model_location_generator)
        output = torch_model(input_tensor)
        logger.info("Inferencing [Torch] Model")
    else:
        performant_runner = EfficientNetb0PerformantRunner(
            device,
            batch_size,
            ttnn.bfloat16,
            ttnn.bfloat16,
            model_location_generator=model_location_generator,
            resolution=(224, 224),
        )
        performant_runner._capture_efficientnetb0_trace_2cqs()
        output = performant_runner.run(torch_input_tensor=input_tensor)

        output = ttnn.to_torch(output)
        logger.info("Inferencing [TTNN] Model")

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, predicted_id = torch.topk(probabilities, 3)

    correct = 0
    predictions = []
    for i in range(batch_size):
        predictions.append(imagenet_label_dict[predicted_id[i].item()])
        logger.info(
            f"Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
        )
        if imagenet_label_dict[labels[i]] == predictions[-1]:
            correct += 1

        accuracy = correct / batch_size * 100
        logger.info(f"Accuracy on the batch: {accuracy:.2f}%")
