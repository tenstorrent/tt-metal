# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from torchvision import transforms, datasets
from loguru import logger

from torch.utils.data import DataLoader
from models.demos.wormhole.mnist.reference.mnist import MnistModel
from models.demos.wormhole.mnist.tt import tt_mnist
from models.utility_functions import disable_persistent_kernel_cache
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_wormhole_b0, skip_for_grayskull


def run_demo_dataset(batch_size, iterations, model_location_generator, mesh_device):
    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    model = MnistModel(state_dict)
    model = model.eval()
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = (2 * batch_size) if mesh_device_flag else batch_size
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=mesh_device,
        )

    correct = 0
    for iters in range(iterations):
        dataloader = DataLoader(test_dataset, batch_size=batch_size)
        x, labels = next(iter(dataloader))
        dataset_predictions = []
        ttnn_predictions = []
        dataset_ttnn_correct = 0
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper, device=mesh_device)
        tt_output = tt_mnist.mnist(mesh_device, batch_size, x, parameters)
        tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
        predicted_probabilities = torch.nn.functional.softmax(tt_output, dim=1)
        _, predicted_label = torch.max(predicted_probabilities, 1)
        tt_output = tt_output
        for i in range(batch_size):
            dataset_predictions.append(labels[i])
            ttnn_predictions.append(predicted_label[i])
            logger.info(f"Iter: {iters} Sample {i}:")
            logger.info(f"Expected Label: {dataset_predictions[i]}")
            logger.info(f"Predicted Label: {ttnn_predictions[i]}")

            if dataset_predictions[i] == ttnn_predictions[i]:
                dataset_ttnn_correct += 1
                correct += 1

        dataset_ttnn_accuracy = dataset_ttnn_correct / (batch_size)
        logger.info(
            f"MNIST Inference Accuracy for iter {iters} of {batch_size} input samples : {dataset_ttnn_accuracy}"
        )

    accuracy = correct / (batch_size * iterations)
    logger.info(f"MNIST Inference Accuracy for {batch_size}x{iterations} Samples : {accuracy}")
    assert accuracy >= 0.96484375, f"Expected accuracy : { 0.96484375} Actual accuracy: {accuracy}"


@skip_for_grayskull()
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("iterations", [1])
def test_demo_dataset(
    batch_size,
    iterations,
    model_location_generator,
    mesh_device,
):
    disable_persistent_kernel_cache()
    return run_demo_dataset(
        batch_size=batch_size,
        iterations=iterations,
        model_location_generator=model_location_generator,
        mesh_device=mesh_device,
    )
