# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch, ttnn
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_alexnet.tt.ttnn_alexnet import ttnn_alexnet
from models.utility_functions import disable_persistent_kernel_cache, disable_compilation_reports
from models.experimental.functional_alexnet.tt.ttnn_alexnet import custom_preprocessor


def run_alexnet(device):
    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: True,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    resize_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    test_input = resize_transform(image).unsqueeze(0)

    ttnn_input = ttnn.from_torch(test_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with torch.inference_mode():
        ttnn_output_tensor = ttnn_alexnet(device, ttnn_input, parameters)
        ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor)
        ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
        ttnn_predicted = ttnn_output_tensor.argmax(-1)

    with torch.inference_mode():
        torch_output_tensor = torch_model(test_input)
        torch_predicted = torch_output_tensor.argmax(-1)

    batch_size = len(test_input)
    correct = 0
    for i in range(batch_size):
        if torch_predicted[i] == ttnn_predicted[i]:
            correct += 1
    accuracy = correct / (batch_size)

    logger.info(f" Accuracy for {batch_size} Samples : {accuracy}")
    logger.info(f"torch_predicted {torch_predicted.squeeze()}")
    logger.info(f"ttnn_predicted {ttnn_predicted.squeeze()}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_alexnet(device):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_alexnet(device)
