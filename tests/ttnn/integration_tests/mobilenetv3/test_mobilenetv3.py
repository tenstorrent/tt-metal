# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from functools import partial
from models.experimental.functional_mobilenetv3.ttnn.ttnn_invertedResidual import (
    InvertedResidualConfig,
)
from models.experimental.functional_mobilenetv3.ttnn.ttnn_mobileNetV3 import ttnn_MobileNetV3
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidual, SElayer, Conv2dNormActivation


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}

        if isinstance(model, MobileNetV3):
            parameters["features"] = {}
            parameters_features = {}
            for index_1, child in enumerate(model.features.children()):
                parameters_features[index_1] = {}
                if isinstance(child, Conv2dNormActivation):
                    parameters_features[index_1][0] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child[0], child[1])
                    parameters_features[index_1][0]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters_features[index_1][0]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )
                elif isinstance(child, InvertedResidual):
                    parameters_features[index_1]["block"] = {}
                    for index, child_1 in enumerate(child.block.children()):
                        parameters_features[index_1]["block"][index] = {}
                        if isinstance(child_1, SElayer):
                            parameters_features[index_1]["block"][index]["fc1"] = {}
                            parameters_features[index_1]["block"][index]["fc1"]["weight"] = ttnn.from_torch(
                                child_1.fc1.weight, dtype=ttnn.bfloat16
                            )
                            parameters_features[index_1]["block"][index]["fc1"]["bias"] = ttnn.from_torch(
                                torch.reshape(child_1.fc1.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                            )
                            parameters_features[index_1]["block"][index]["fc2"] = {}
                            parameters_features[index_1]["block"][index]["fc2"]["weight"] = ttnn.from_torch(
                                child_1.fc2.weight, dtype=ttnn.bfloat16
                            )
                            parameters_features[index_1]["block"][index]["fc2"]["bias"] = ttnn.from_torch(
                                torch.reshape(child_1.fc2.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                            )
                        elif isinstance(child_1, Conv2dNormActivation):
                            parameters_features[index_1]["block"][index][0] = {}
                            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_1[0], child_1[1])
                            parameters_features[index_1]["block"][index][0]["weight"] = ttnn.from_torch(
                                conv_weight, dtype=ttnn.bfloat16
                            )
                            parameters_features[index_1]["block"][index][0]["bias"] = ttnn.from_torch(
                                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                            )
            parameters["features"] = parameters_features
            parameters["classifier"] = {}
            parameters["classifier"][0] = {}
            parameters["classifier"][0]["weight"] = preprocess_linear_weight(
                model.classifier[0].weight, dtype=ttnn.bfloat16
            )
            parameters["classifier"][0]["bias"] = preprocess_linear_bias(model.classifier[0].bias, dtype=ttnn.bfloat16)

            parameters["classifier"][3] = {}
            parameters["classifier"][3]["weight"] = preprocess_linear_weight(
                model.classifier[3].weight, dtype=ttnn.bfloat16
            )
            parameters["classifier"][3]["bias"] = preprocess_linear_bias(model.classifier[3].bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size,channels,height,width",
    [
        (1, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_mobilenetv3(device, reset_seeds, batch_size, channels, height, width):
    torch_input_tensor = torch.randn(batch_size, channels, height, width)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )

    from torchvision import models

    mobilenet = models.mobilenet_v3_small(weights=True)
    torch_model = mobilenet

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
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    ]

    last_channel = adjust_channels(1024 // reduce_divider)

    ttnn_model = ttnn_MobileNetV3(
        inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, parameters=parameters
    )

    ttnn_output_tensor = ttnn_model(device, ttnn_input_tensor)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    assert_with_pcc(ttnn_output_tensor, torch_output_tensor, 0.99)  # 0.83
