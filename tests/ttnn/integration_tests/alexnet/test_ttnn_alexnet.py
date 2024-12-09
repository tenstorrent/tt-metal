# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch, ttnn
import torch.nn as nn
from loguru import logger
from torchvision import models
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.alexnet.tt.ttnn_alexnet import ttnn_alexnet
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.alexnet.tt.ttnn_alexnet import custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [(torch.rand((2, 3, 64, 64))), (torch.rand((4, 3, 64, 64))), (torch.rand((8, 3, 64, 64)))],
    ids=["input_tensor1", "input_tensor2", "input_tensor3"],
)
def test_alexnet(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: True,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with torch.inference_mode():
        ttnn_output_tensor = ttnn_alexnet(device, ttnn_input, parameters)
        ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    with torch.inference_mode():
        torch_output_tensor = torch_model(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_output_tensor, torch_output_tensor, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
