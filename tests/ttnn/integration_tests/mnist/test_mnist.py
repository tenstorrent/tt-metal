# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.mnist.reference.mnist import MnistModel
from models.demos.mnist.tt import tt_mnist


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [128],
)
def test_mnist(reset_seeds, device, batch_size, model_location_generator):
    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    model = MnistModel(state_dict)
    model = model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    x, labels = next(iter(dataloader))

    torch_output = model(x)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: True,
        device=device,
    )
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)

    tt_output = tt_mnist.mnist(device, batch_size, x, parameters)

    tt_output = ttnn.to_torch(tt_output)
    print(tt_output.shape)
    assert_with_pcc(torch_output, tt_output, 0.99)
