# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.mnist_like_model.reference import mnist_like_model
from models.experimental.mnist_like_model.ttnn import ttnn_mnist_like_model
from models.experimental.mnist_like_model.ttnn.model_preprocessing import (
    create_mnist_like_model_input_tensors,
    create_mnist_like_model_model_parameters,
)
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_mnist_like_model(device, reset_seeds):
    torch_model = mnist_like_model.MnistLikeModel(11)
    torch_model.eval()
    torch_input, ttnn_input = create_mnist_like_model_input_tensors()
    torch_output = torch_model(torch_input)
    parameters = create_mnist_like_model_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_mnist_like_model.TtMnistLikeModel(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)  # PCC = 0.99
    logger.info(pcc_message)
