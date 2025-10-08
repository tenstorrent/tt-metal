# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from models.experimental.mobileNetV3.tt.ttnn_invertedResidual import (
    ttnn_InvertedResidual,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from models.experimental.mobileNetV3.tests.pcc.common import inverted_residual_setting


@pytest.mark.parametrize(
    "batch_size,channels,height,width,feature_i",
    [
        (1, 16, 112, 112, 1),
        (1, 16, 56, 56, 2),
        (1, 24, 28, 28, 3),
        (1, 24, 28, 28, 4),
        (1, 40, 14, 14, 5),
        (1, 40, 14, 14, 6),
        (1, 40, 14, 14, 7),
        (1, 48, 14, 14, 8),
        (1, 48, 14, 14, 9),
        (1, 96, 7, 7, 10),
        (1, 96, 7, 7, 11),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_invertedResidual(device, reset_seeds, batch_size, channels, height, width, feature_i):
    torch_input_tensor = torch.randn(batch_size, channels, height, width)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )

    mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    torch_model = mobilenet.features[feature_i]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    torch_output_tensor = torch_model(torch_input_tensor)

    ttnn_model = ttnn_InvertedResidual(inverted_residual_setting[feature_i - 1], parameters=parameters)

    ttnn_output_tensor = ttnn_model(device, ttnn_input_tensor)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
    ttnn_output_tensor = torch.permute(ttnn_output_tensor, (0, 3, 1, 2))

    assert_with_pcc(ttnn_output_tensor, torch_output_tensor, 0.99)
