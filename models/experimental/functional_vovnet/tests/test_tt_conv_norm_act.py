# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger
import ttnn
from models.experimental.functional_vovnet.tt.conv_norm_act import TtConvNormAct
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
import torch.nn as nn


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        weight = model.weight
        if model.bias is not None:
            bias = model.bias
        else:
            bias = None
        # while weight.dim() < 4:
        #     weight = weight.unsqueeze(0)
        if bias is not None:
            while bias.dim() < 4:
                bias = bias.unsqueeze(0)
            parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.bfloat16)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16)
    elif isinstance(model, nn.BatchNorm2d):
        parameters["weight"] = preprocess_conv_parameter(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_conv_parameter(model.bias, dtype=ttnn.bfloat16)
        parameters["running_mean"] = preprocess_conv_parameter(model.running_mean, dtype=ttnn.bfloat16)
        parameters["running_var"] = preprocess_conv_parameter(model.running_var, dtype=ttnn.bfloat16)
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnet_conv_norm_act_inference(device, reset_seeds):
    base_address = f"stem[0]"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=custom_preprocessor,
    )
    # print(parameters[stem.0.conv.weight.shape])

    torch_model = model.stem[0]
    tt_model = TtConvNormAct(
        stride=2,
        base_address=base_address,
        device=device,
        # torch_model=model.state_dict(),
        torch_model=parameters.stem[0],
    )

    # run torch model
    input = torch.rand(1, 3, 224, 224)
    model_output = torch_model(input)

    # run tt model
    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output[0])
    assert_with_pcc(model_output, tt_output_torch, 0.99)
