# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from functools import partial
from models.experimental.functional_mobilenetv3.ttnn.ttnn_invertedResidual import (
    ttnn_InvertedResidual,
    InvertedResidualConfig,
)
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from torchvision.models.mobilenetv3 import InvertedResidual, SElayer, Conv2dNormActivation


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, InvertedResidual):
            for index, child in enumerate(model.block.children()):
                parameters[index] = {}
                if isinstance(child, SElayer):
                    parameters[index]["fc1"] = {}
                    parameters[index]["fc1"]["weight"] = ttnn.from_torch(child.fc1.weight, dtype=ttnn.bfloat16)
                    parameters[index]["fc1"]["bias"] = ttnn.from_torch(
                        torch.reshape(child.fc1.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )
                    parameters[index]["fc2"] = {}
                    parameters[index]["fc2"]["weight"] = ttnn.from_torch(child.fc2.weight, dtype=ttnn.bfloat16)
                    parameters[index]["fc2"]["bias"] = ttnn.from_torch(
                        torch.reshape(child.fc2.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )
                elif isinstance(child, Conv2dNormActivation):
                    parameters[index][0] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child[0], child[1])
                    parameters[index][0]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[index][0]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )
        return parameters

    return custom_preprocessor


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

    from torchvision import models
    from torchvision.models.mobilenetv3 import SElayer

    mobilenet = models.mobilenet_v3_small(weights=True)
    torch_model = mobilenet.features[feature_i]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    torch_output_tensor = torch_model(torch_input_tensor)

    reduce_divider = 1
    dilation = 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=1.0)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),
        bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    ]

    ttnn_model = ttnn_InvertedResidual(inverted_residual_setting[feature_i - 1], parameters=parameters)

    ttnn_output_tensor = ttnn_model(device, ttnn_input_tensor)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
    ttnn_output_tensor = torch.permute(ttnn_output_tensor, (0, 3, 1, 2))

    assert_with_pcc(ttnn_output_tensor, torch_output_tensor, 0.99)
