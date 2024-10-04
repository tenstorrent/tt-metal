# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.lenet.tt import tt_lenet
from models.demos.lenet import lenet_utils


@pytest.mark.parametrize(
    "batch_size",
    [8],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_lenet_inference(device, batch_size, model_location_generator, reset_seeds):
    num_classes = 10
    test_input, images, outputs = lenet_utils.get_test_data(batch_size)

    pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
    torch_LeNet, state_dict = lenet_utils.load_torch_lenet(pt_model_path, num_classes)
    model = torch_LeNet.float()
    model = torch_LeNet.eval()

    torch_output = model(test_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_LeNet,
        custom_preprocessor=lenet_utils.custom_preprocessor,
    )
    parameters = lenet_utils.custom_preprocessor_device(parameters, device)
    x = test_input
    x = test_input.permute(0, 2, 3, 1)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
    tt_output = tt_lenet.Lenet(x, model, batch_size, num_classes, device, parameters, reset_seeds)

    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.99)
