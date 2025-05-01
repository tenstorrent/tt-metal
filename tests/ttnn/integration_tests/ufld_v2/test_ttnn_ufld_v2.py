# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os
import pytest
import torch
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
    infer_ttnn_module_args,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34, BasicBlock
from models.demos.ufld_v2.ttnn.ttnn_ufld_v2 import TtnnUFLDv2
from models.demos.ufld_v2.ttnn.ttnn_basic_block import TtnnBasicBlock
from tests.ttnn.utils_for_testing import assert_with_pcc


def custom_preprocessor_basic_block(model, name):
    parameters = {}
    if isinstance(model, BasicBlock):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
        parameters["conv2"] = {}
        parameters["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


def custom_preprocessor_whole_model(model, name):
    parameters = {}
    if isinstance(model, TuSimple34):
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.conv1, model.res_model.bn1)
        parameters["res_model"] = {}
        parameters["res_model"]["conv1"] = {}
        parameters["res_model"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer1[0].conv1, model.res_model.layer1[0].bn1)
        parameters["res_model"]["layer1_0"] = {}
        parameters["res_model"]["layer1_0"]["conv1"] = {}
        parameters["res_model"]["layer1_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer1[0].conv2, model.res_model.layer1[0].bn2)
        parameters["res_model"]["layer1_0"]["conv2"] = {}
        parameters["res_model"]["layer1_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer1[1].conv1, model.res_model.layer1[1].bn1)
        parameters["res_model"]["layer1_1"] = {}
        parameters["res_model"]["layer1_1"]["conv1"] = {}
        parameters["res_model"]["layer1_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer1[1].conv2, model.res_model.layer1[1].bn2)
        parameters["res_model"]["layer1_1"]["conv2"] = {}
        parameters["res_model"]["layer1_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer1[2].conv1, model.res_model.layer1[2].bn1)
        parameters["res_model"]["layer1_2"] = {}
        parameters["res_model"]["layer1_2"]["conv1"] = {}
        parameters["res_model"]["layer1_2"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_2"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer1[2].conv2, model.res_model.layer1[2].bn2)
        parameters["res_model"]["layer1_2"]["conv2"] = {}
        parameters["res_model"]["layer1_2"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer1_2"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer2[0].conv1, model.res_model.layer2[0].bn1)
        parameters["res_model"]["layer2_0"] = {}
        parameters["res_model"]["layer2_0"]["conv1"] = {}
        parameters["res_model"]["layer2_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer2[0].conv2, model.res_model.layer2[0].bn2)
        parameters["res_model"]["layer2_0"]["conv2"] = {}
        parameters["res_model"]["layer2_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer2[1].conv1, model.res_model.layer2[1].bn1)
        parameters["res_model"]["layer2_1"] = {}
        parameters["res_model"]["layer2_1"]["conv1"] = {}
        parameters["res_model"]["layer2_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer2[1].conv2, model.res_model.layer2[1].bn2)
        parameters["res_model"]["layer2_1"]["conv2"] = {}
        parameters["res_model"]["layer2_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer2[2].conv1, model.res_model.layer2[2].bn1)
        parameters["res_model"]["layer2_2"] = {}
        parameters["res_model"]["layer2_2"]["conv1"] = {}
        parameters["res_model"]["layer2_2"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_2"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer2[2].conv2, model.res_model.layer2[2].bn2)
        parameters["res_model"]["layer2_2"]["conv2"] = {}
        parameters["res_model"]["layer2_2"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_2"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer2[3].conv1, model.res_model.layer2[3].bn1)
        parameters["res_model"]["layer2_3"] = {}
        parameters["res_model"]["layer2_3"]["conv1"] = {}
        parameters["res_model"]["layer2_3"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_3"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer2[3].conv2, model.res_model.layer2[3].bn2)
        parameters["res_model"]["layer2_3"]["conv2"] = {}
        parameters["res_model"]["layer2_3"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer2_3"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        if hasattr(model.res_model.layer2[0], "downsample") and model.res_model.layer2[0].downsample is not None:
            downsample = model.res_model.layer2[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["res_model"]["layer2_0"]["downsample"] = {}
                parameters["res_model"]["layer2_0"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.float32
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["res_model"]["layer2_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[0].conv1, model.res_model.layer3[0].bn1)
        parameters["res_model"]["layer3_0"] = {}
        parameters["res_model"]["layer3_0"]["conv1"] = {}
        parameters["res_model"]["layer3_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[0].conv2, model.res_model.layer3[0].bn2)
        parameters["res_model"]["layer3_0"]["conv2"] = {}
        parameters["res_model"]["layer3_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[1].conv1, model.res_model.layer3[1].bn1)
        parameters["res_model"]["layer3_1"] = {}
        parameters["res_model"]["layer3_1"]["conv1"] = {}
        parameters["res_model"]["layer3_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[1].conv2, model.res_model.layer3[1].bn2)
        parameters["res_model"]["layer3_1"]["conv2"] = {}
        parameters["res_model"]["layer3_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[2].conv1, model.res_model.layer3[2].bn1)
        parameters["res_model"]["layer3_2"] = {}
        parameters["res_model"]["layer3_2"]["conv1"] = {}
        parameters["res_model"]["layer3_2"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_2"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[2].conv2, model.res_model.layer3[2].bn2)
        parameters["res_model"]["layer3_2"]["conv2"] = {}
        parameters["res_model"]["layer3_2"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_2"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[3].conv1, model.res_model.layer3[3].bn1)
        parameters["res_model"]["layer3_3"] = {}
        parameters["res_model"]["layer3_3"]["conv1"] = {}
        parameters["res_model"]["layer3_3"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_3"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[3].conv2, model.res_model.layer3[3].bn2)
        parameters["res_model"]["layer3_3"]["conv2"] = {}
        parameters["res_model"]["layer3_3"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_3"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[4].conv1, model.res_model.layer3[4].bn1)
        parameters["res_model"]["layer3_4"] = {}
        parameters["res_model"]["layer3_4"]["conv1"] = {}
        parameters["res_model"]["layer3_4"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_4"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[4].conv2, model.res_model.layer3[4].bn2)
        parameters["res_model"]["layer3_4"]["conv2"] = {}
        parameters["res_model"]["layer3_4"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_4"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[5].conv1, model.res_model.layer3[5].bn1)
        parameters["res_model"]["layer3_5"] = {}
        parameters["res_model"]["layer3_5"]["conv1"] = {}
        parameters["res_model"]["layer3_5"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_5"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer3[5].conv2, model.res_model.layer3[5].bn2)
        parameters["res_model"]["layer3_5"]["conv2"] = {}
        parameters["res_model"]["layer3_5"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer3_5"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        if hasattr(model.res_model.layer3[0], "downsample") and model.res_model.layer3[0].downsample is not None:
            downsample = model.res_model.layer3[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["res_model"]["layer3_0"]["downsample"] = {}
                parameters["res_model"]["layer3_0"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.float32
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["res_model"]["layer3_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer4[0].conv1, model.res_model.layer4[0].bn1)
        parameters["res_model"]["layer4_0"] = {}
        parameters["res_model"]["layer4_0"]["conv1"] = {}
        parameters["res_model"]["layer4_0"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_0"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer4[0].conv2, model.res_model.layer4[0].bn2)
        parameters["res_model"]["layer4_0"]["conv2"] = {}
        parameters["res_model"]["layer4_0"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_0"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer4[1].conv1, model.res_model.layer4[1].bn1)
        parameters["res_model"]["layer4_1"] = {}
        parameters["res_model"]["layer4_1"]["conv1"] = {}
        parameters["res_model"]["layer4_1"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_1"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer4[1].conv2, model.res_model.layer4[1].bn2)
        parameters["res_model"]["layer4_1"]["conv2"] = {}
        parameters["res_model"]["layer4_1"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_1"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer4[2].conv1, model.res_model.layer4[2].bn1)
        parameters["res_model"]["layer4_2"] = {}
        parameters["res_model"]["layer4_2"]["conv1"] = {}
        parameters["res_model"]["layer4_2"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_2"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.layer4[2].conv2, model.res_model.layer4[2].bn2)
        parameters["res_model"]["layer4_2"]["conv2"] = {}
        parameters["res_model"]["layer4_2"]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["layer4_2"]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        if hasattr(model.res_model.layer4[0], "downsample") and model.res_model.layer4[0].downsample is not None:
            downsample = model.res_model.layer4[0].downsample
            if isinstance(downsample, torch.nn.Sequential):
                conv_layer = downsample[0]
                bn_layer = downsample[1]
                weight, bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters["res_model"]["layer4_0"]["downsample"] = {}
                parameters["res_model"]["layer4_0"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.float32
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["res_model"]["layer4_0"]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        parameters["pool"] = {}
        parameters["pool"]["weight"] = ttnn.from_torch(model.pool.weight, dtype=ttnn.float32)
        if model.pool.bias is not None:
            bias = model.pool.bias.reshape((1, 1, 1, -1))
            parameters["pool"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        else:
            parameters["pool"]["bias"] = None

        parameters["cls"] = {}
        parameters["cls"]["linear_1"] = {}
        parameters["cls"]["linear_1"]["weight"] = preprocess_linear_weight(model.cls[1].weight, dtype=ttnn.bfloat16)
        if model.cls[1].bias is not None:
            parameters["cls"]["linear_1"]["bias"] = preprocess_linear_bias(model.cls[1].bias, dtype=ttnn.bfloat16)
        else:
            parameters["cls"]["linear_1"]["bias"] = None

        parameters["cls"]["linear_2"] = {}
        parameters["cls"]["linear_2"]["weight"] = preprocess_linear_weight(model.cls[3].weight, dtype=ttnn.bfloat16)
        if model.cls[3].bias is not None:
            parameters["cls"]["linear_2"]["bias"] = preprocess_linear_bias(model.cls[3].bias, dtype=ttnn.bfloat16)
        else:
            parameters["cls"]["linear_2"]["bias"] = None

    return parameters


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 64, 80, 200),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ufld_v2_basic_block(device, batch_size, input_channels, height, width):
    torch_model = TuSimple34(input_height=height, input_width=width).res_model.layer1[0]
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width), dtype=torch.bfloat16)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        (ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2]),
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor_basic_block,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnBasicBlock(parameters.conv_args, parameters, device=device)
    torch_out = torch_model(torch_input_tensor)
    ttnn_output = ttnn_model(ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_out.shape)
    assert_with_pcc(ttnn_output, torch_out, 0.99)


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        #  True
    ],  # uncomment  to run the model for real weights
    ids=[
        "pretrained_weight_false",
        # "pretrained_weight_true",  # uncomment to run the model for real weights
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ufld_v2_model(device, batch_size, input_channels, height, width, use_pretrained_weight):
    torch_model = TuSimple34(input_height=height, input_width=width)
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width), dtype=torch.bfloat16)
    torch_output = torch_model(torch_input_tensor)
    if use_pretrained_weight:
        weights_path = "models/demos/ufld_v2/tusimple_res34.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/ufld_v2/weights_download.sh")
            state_dict = torch.load(weights_path)
            new_state_dict = {}
            for key, value in state_dict["model"].items():
                new_key = key.replace("model.", "res_model.")
                new_state_dict[new_key] = value
            torch_model.load_state_dict(new_state_dict)

    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        (ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2]),
        ttnn_input_tensor.shape[3],
    )

    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor_whole_model,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnUFLDv2(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    torch_output, pred_list = torch_model(torch_input_tensor)
    ttnn_output = ttnn_model(input=ttnn_input_tensor, batch_size=batch_size)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(dim=0).squeeze(dim=0)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
