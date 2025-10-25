# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.mobileNetV3.tt.ttnn_squeezeExcitation import ttnn_SqueezeExcitation
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from torchvision import models
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_squeezeExcitation(device, reset_seeds):
    torch_input_tensor = torch.randn(1, 16, 56, 56)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    mobilenet = models.mobilenet_v3_small(weights=True)
    torch_model = mobilenet.features[1].block[1]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    torch_output_tensor = torch_model(torch_input_tensor)

    ttnn_model = ttnn_SqueezeExcitation(16, 8, ttnn.relu, ttnn.hardsigmoid, parameters)

    ttnn_output_tensor = ttnn_model(device, ttnn_input_tensor)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
    ttnn_output_tensor = torch.permute(ttnn_output_tensor, (0, 3, 1, 2))

    assert_with_pcc(ttnn_output_tensor, torch_output_tensor, 0.99)
