# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest
import ttnn
from loguru import logger
from models.experimental.functional_mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.experimental.functional_mobilenetv2.tt import ttnn_mobilenetv2
from models.experimental.functional_mobilenetv2.demo.demo_utils import get_data_loader, get_batch
from models.experimental.functional_mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_model_parameters,
)
from models.utility_functions import skip_for_grayskull


def mobilenetv2_demo(
    device, reset_seeds, model_location_generator, imagenet_label_dict, iterations, batch_size, use_pretrained_weight
):
    weights_path = "models/experimental/functional_mobilenetv2/mobilenet_v2-b0353104.pth"
    if not os.path.exists(weights_path):
        os.system("bash models/experimental/functional_mobilenetv2/weights_download.sh")
    if use_pretrained_weight:
        state_dict = torch.load(weights_path)
        ds_state_dict = {k: v for k, v in state_dict.items()}

        torch_model = Mobilenetv2()
        new_state_dict = {
            name1: parameter2
            for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
            if isinstance(parameter2, torch.FloatTensor)
        }
        torch_model.load_state_dict(new_state_dict)
    else:
        torch_model = Mobilenetv2()
        state_dict = torch_model.state_dict()

    torch_model.eval()

    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_model = ttnn_mobilenetv2.MobileNetV2(parameters, device, batchsize=batch_size)
    correct = 0

    for iter in range(iterations):
        predictions = []
        inputs, labels = get_batch(data_loader)
        torch_input_tensor = torch.unsqueeze(inputs, 0)
        torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

        ttnn_input_tensor = torch_input_tensor.reshape(
            1,
            1,
            torch_input_tensor.shape[0] * torch_input_tensor.shape[1] * torch_input_tensor.shape[2],
            torch_input_tensor.shape[3],
        )

        ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)
        tt_output = ttnn_model(ttnn_input_tensor)
        tt_output = ttnn.from_device(tt_output, blocking=True).to_torch().to(torch.float)
        prediction = tt_output.argmax(dim=-1)

        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1

        del tt_output, inputs, labels, predictions

    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        True,
    ],
    ids=[
        "pretrained_weight_false",
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "batch_size, iterations",
    [
        [1, 100],
    ],
)
def test_mobilenetv2_demo(
    device, reset_seeds, iterations, batch_size, use_pretrained_weight, model_location_generator, imagenet_label_dict
):
    mobilenetv2_demo(
        device,
        reset_seeds,
        model_location_generator,
        imagenet_label_dict,
        iterations,
        batch_size,
        use_pretrained_weight,
    )
