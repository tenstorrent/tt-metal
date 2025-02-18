# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import torch.nn as nn
from models.experimental.functional_Ultralane_detection_V2.tt.ttnn_Ultralane_fast_detection_v2 import (
    ttnn_Resnet_34,
)
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d, infer_ttnn_module_args
from models.experimental.functional_Ultralane_detection_V2.reference.tu_simple_model import Tu_Simple
from tests.ttnn.utils_for_testing import assert_with_pcc


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        else:
            parameters["bias"] = None

    return parameters


def p(x, b="x"):
    print(f"{b}'s shape is {x.shape}")
    print(f"{b}'s layout is {x.layout}")
    print(f"{b}'s dtype is {x.dtype}")
    print(f"{b}'s config is {x.memory_config()}")


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_tu_simple_res34(device, batch_size, input_channels, height, width):
    torch_model = Tu_Simple(input_height=height, input_width=width).model
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width), dtype=torch.bfloat16)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=None
    )

    print("params are", parameters.conv_args)
    ttnn_model = ttnn_Resnet_34(conv_args=torch_model, conv_pth=parameters, device=device)
    print("input tt tesnor")
    p(ttnn_input_tensor, "ttnn_input_tensor")
    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)
    # l1 = torch.load("/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_Ultralane_detection_V2/dumps/torch_out.pth")
    # l2 = torch.load("/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_Ultralane_detection_V2/dumps/ttnn_out.pth")
    # assert_with_pcc(l1,l2,1.0)
