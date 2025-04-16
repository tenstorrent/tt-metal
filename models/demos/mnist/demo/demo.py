# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.mnist.reference.mnist import MnistModel
from models.demos.mnist.tt import tt_mnist
from models.utility_functions import disable_persistent_kernel_cache


def run_demo_dataset(device, batch_size, iterations, model_location_generator):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    model = MnistModel(state_dict)
    model = model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: True,
        device=device,
    )
    correct = 0
    for iters in range(iterations):
        dataloader = DataLoader(test_dataset, batch_size=batch_size)
        x, labels = next(iter(dataloader))
        dataset_predictions = []
        ttnn_predictions = []
        dataset_ttnn_correct = 0
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
        tt_output = tt_mnist.mnist(device, batch_size, x, parameters)
        tt_output = ttnn.to_torch(tt_output)
        predicted_probabilities = torch.nn.functional.softmax(tt_output, dim=1)
        _, predicted_label = torch.max(predicted_probabilities, 1)
        tt_output = tt_output
        for i in range(batch_size):
            dataset_predictions.append(labels[i])
            ttnn_predictions.append(predicted_label[i])
            logger.info(f"Iter: {iters} Sample {i}:")
            logger.info(f"Expected Label: {dataset_predictions[i]}")
            logger.info(f"TT Predicted Label: {ttnn_predictions[i]}")
            if dataset_predictions[i] == ttnn_predictions[i]:
                dataset_ttnn_correct += 1
                correct += 1
        dataset_ttnn_accuracy = dataset_ttnn_correct / (batch_size)
        logger.info(
            f"ImageNet Inference Accuracy for iter {iters} of {batch_size} input samples : {dataset_ttnn_accuracy}"
        )
    accuracy = correct / (batch_size * iterations)
    logger.info(f"ImageNet Inference Accuracy for {batch_size}x{iterations} Samples : {accuracy}")
    assert accuracy >= 0.96875, f"Expected accuracy : {0.96875} Actual accuracy: {accuracy}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("iterations", [1])
def test_demo_dataset(
    device,
    batch_size,
    iterations,
    model_location_generator,
):
    disable_persistent_kernel_cache()

    return run_demo_dataset(
        device=device,
        batch_size=batch_size,
        iterations=iterations,
        model_location_generator=model_location_generator,
    )
