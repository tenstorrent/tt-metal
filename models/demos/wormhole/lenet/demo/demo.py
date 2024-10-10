# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from torchvision import transforms, datasets
from loguru import logger

from torch.utils.data import DataLoader

from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.lenet.tt import tt_lenet
from models.demos.wormhole.lenet import lenet_utils
from models.utility_functions import is_wormhole_b0, skip_for_grayskull


def run_demo_dataset(mesh_device, batch_size, iterations, model_location_generator, reset_seeds):
    num_classes = 10
    test_input, images, outputs = lenet_utils.get_test_data(batch_size)

    pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
    torch_LeNet, state_dict = lenet_utils.load_torch_lenet(pt_model_path, num_classes)
    model = torch_LeNet.float()
    model = torch_LeNet.eval()
    torch_output = model(test_input)

    inputs_mesh_mapper = None
    weights_mesh_mapper = None
    output_mesh_composer = None

    if is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

        with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
            parameters = preprocess_model_parameters(
                initialize_model=lambda: model, custom_preprocessor=lenet_utils.custom_preprocessor
            )

    else:
        if is_wormhole_b0():
            mesh_device = ttnn.open_device(device_id=0)
            parameters = preprocess_model_parameters(
                initialize_model=lambda: model, custom_preprocessor=lenet_utils.custom_preprocessor
            )

    parameters = lenet_utils.custom_preprocessor_device(parameters, device=mesh_device)
    correct = 0
    for iters in range(iterations):
        x = test_input.permute(0, 2, 3, 1)
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)
        tt_output = tt_lenet.Lenet(
            x,
            model,
            batch_size,
            num_classes,
            mesh_device,
            parameters,
            reset_seeds,
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )
        tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
        _, torch_predicted = torch.max(torch_output.data, -1)
        _, ttnn_predicted = torch.max(tt_output.data, -1)

        for i in range(batch_size):
            logger.info(f"Iter: {iters} Sample {i}:")
            logger.info(f"torch Label: {torch_predicted[i]}")
            logger.info(f"Predicted Label: {ttnn_predicted[i]}")

            if torch_predicted[i] == ttnn_predicted[i]:
                correct += 1

    accuracy = correct / (batch_size * iterations)
    logger.info(f"ImageNet Inference Accuracy for {batch_size}x{iterations} Samples : {accuracy}")


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("iterations", [1])
def test_demo_dataset(
    mesh_device,
    batch_size,
    iterations,
    model_location_generator,
    reset_seeds,
):
    return run_demo_dataset(
        reset_seeds=reset_seeds,
        mesh_device=mesh_device,
        batch_size=batch_size,
        iterations=iterations,
        model_location_generator=model_location_generator,
    )
