# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
from models.experimental.functional_UFLD_v2.ttnn.ttnn_UFLD_v2 import (
    ttnn_UFLD_V2,
)
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
from models.experimental.functional_UFLD_v2.reference.UFLD_v2_model import Tu_Simple
from tests.ttnn.utils_for_testing import assert_with_pcc


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, Tu_Simple):
        # conv1,bn1
        weight, bias = fold_batch_norm2d_into_conv2d(model.res_model.conv1, model.res_model.bn1)
        parameters["res_model"] = {}
        parameters["res_model"]["conv1"] = {}
        parameters["res_model"]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["res_model"]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        # layer0 - 0
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
        # layer1 - 1
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
        # layer1 - 2
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
        # layer-2-0
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
        # layer2 - 1
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
        # layer2-2
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
        # layer2-3
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
        # downsample layer2[0]
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
        # layer3-0
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
        # layer3-1
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
        # layer3-2
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
        # layer3-3
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
        # layer3-4
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
        # layer3-5
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
        # downsample - layer3[0]
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
        # layer4-0
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
        # layer4 - 1
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
        # layer4-2
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
        # downsample - layer3[0]
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

        # pool
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
        (1, 3, 320, 800),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_UFD_V2_Model(device, batch_size, input_channels, height, width):
    torch_model = Tu_Simple(input_height=height, input_width=width)
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width), dtype=torch.bfloat16)
    torch_output = torch_model(torch_input_tensor)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_model = ttnn_UFLD_V2(conv_args=torch_model, conv_pth=parameters, device=device)
    torch_output, pred_list = torch_model(torch_input_tensor)
    ttnn_output, tt_pred_list = ttnn_model(input=ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output)
    tt_pred_list["loc_row"] = ttnn.to_torch(tt_pred_list["loc_row"])
    tt_pred_list["loc_col"] = ttnn.to_torch(tt_pred_list["loc_col"])
    tt_pred_list["exist_row"] = ttnn.to_torch(tt_pred_list["exist_row"])
    tt_pred_list["exist_col"] = ttnn.to_torch(tt_pred_list["exist_col"])
    assert_with_pcc(torch_output, ttnn_output, 0.999)
    assert_with_pcc(tt_pred_list["loc_row"], pred_list["loc_row"], 0.999)
    assert_with_pcc(tt_pred_list["loc_col"], pred_list["loc_col"], 0.999)
    assert_with_pcc(tt_pred_list["exist_row"], pred_list["exist_row"], 0.999)
    assert_with_pcc(tt_pred_list["exist_col"], pred_list["exist_col"], 0.999)
