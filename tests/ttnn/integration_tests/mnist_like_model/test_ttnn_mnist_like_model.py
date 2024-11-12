# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_mnist_like_model.reference import mnist_like_model
from models.experimental.functional_mnist_like_model.ttnn import ttnn_mnist_like_model
from models.experimental.functional_mnist_like_model.ttnn.model_preprocessing import (
    create_mnist_like_model_input_tensors,
    create_mnist_like_model_model_parameters,
)
import os
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_mnist_like_model(device, reset_seeds):
    state_dict = torch.load("models/experimental/functional_mnist_like_model/conv_mnist.pth")
    ds_state_dict = {k: v for k, v in state_dict.items()}
    torch_model = mnist_like_model.Mnist_like_model(11)
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    torch_input, ttnn_input = create_mnist_like_model_input_tensors()
    torch_output = torch_model(torch_input)
    parameters = create_mnist_like_model_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_mnist_like_model.Mnist_like_model(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)

    # N,C,H,W = torch_output.shape
    # ttnn_output = ttnn.to_torch(ttnn_output)
    # ttnn_output = ttnn_output.reshape(N, H, W, C )
    # ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)  # PCC = 0.99
    logger.info(pcc_message)
