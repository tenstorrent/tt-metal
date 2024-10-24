# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.mnist.reference.mnist import MnistModel
from models.utility_functions import is_wormhole_b0, skip_for_grayskull
from models.demos.wormhole.mnist.tt import tt_mnist
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size",
    [4],
)
def test_mnist(mesh_device, reset_seeds, batch_size, model_location_generator):
    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    model = MnistModel(state_dict)
    model = model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    x, labels = next(iter(dataloader))
    torch_output = model(x)
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    mesh_device_flag = True
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(initialize_model=lambda: model, device=mesh_device)

    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper, device=mesh_device)
    tt_output = tt_mnist.mnist(mesh_device, batch_size, x, parameters)
    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output, tt_output, 0.99)
