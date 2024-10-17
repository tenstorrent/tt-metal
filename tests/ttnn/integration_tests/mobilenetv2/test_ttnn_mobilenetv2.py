# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.experimental.functional_mobilenetv2.tt.ttnn_mobilenetv2 import TtMobilenetv2

import ttnn
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d
import pytest


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, Mobilenetv2):
            ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1["activation"] = "relu"

            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
            update_ttnn_module_args(ttnn_module_args.c1)
            parameters["c1"], c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
            )

            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
            parameters["c2"] = {}
            parameters["c2"]["weight"] = conv2_weight
            parameters["c2"]["bias"] = conv2_bias

            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
            parameters["c3"] = {}
            parameters["c3"]["weight"] = conv3_weight
            parameters["c3"]["bias"] = conv3_bias

            ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4["activation"] = "relu"

            conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
            update_ttnn_module_args(ttnn_module_args.c4)
            parameters["c4"], c4_parallel_config = preprocess_conv2d(
                conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
            )

            conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
            parameters["c5"] = {}
            parameters["c5"]["weight"] = conv5_weight
            parameters["c5"]["bias"] = conv5_bias

            conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
            parameters["c6"] = {}
            parameters["c6"]["weight"] = conv6_weight
            parameters["c6"]["bias"] = conv6_bias

            conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
            parameters["c7"] = {}
            parameters["c7"]["weight"] = conv7_weight
            parameters["c7"]["bias"] = conv7_bias

            conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
            parameters["c8"] = {}
            parameters["c8"]["weight"] = conv8_weight
            parameters["c8"]["bias"] = conv8_bias

            conv9_weight, conv9_bias = fold_batch_norm2d_into_conv2d(model.c9, model.b9)
            parameters["c9"] = {}
            parameters["c9"]["weight"] = conv9_weight
            parameters["c9"]["bias"] = conv9_bias

            conv10_weight, conv10_bias = fold_batch_norm2d_into_conv2d(model.c10, model.b10)
            parameters["c10"] = {}
            parameters["c10"]["weight"] = conv10_weight
            parameters["c10"]["bias"] = conv10_bias

            conv11_weight, conv11_bias = fold_batch_norm2d_into_conv2d(model.c11, model.b11)
            parameters["c11"] = {}
            parameters["c11"]["weight"] = conv11_weight
            parameters["c11"]["bias"] = conv11_bias

            ttnn_module_args.c12["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c12["use_shallow_conv_variant"] = False

            conv12_weight, conv12_bias = fold_batch_norm2d_into_conv2d(model.c12, model.b12)
            update_ttnn_module_args(ttnn_module_args.c12)
            parameters["c12"], c12_parallel_config = preprocess_conv2d(
                conv12_weight, conv12_bias, ttnn_module_args.c12, return_parallel_config=True
            )

            ttnn_module_args.c13["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c13["activation"] = "relu"

            conv13_weight, conv13_bias = fold_batch_norm2d_into_conv2d(model.c13, model.b13)
            update_ttnn_module_args(ttnn_module_args.c13)
            parameters["c13"], c13_parallel_config = preprocess_conv2d(
                conv13_weight, conv13_bias, ttnn_module_args.c13, return_parallel_config=True
            )

            conv14_weight, conv14_bias = fold_batch_norm2d_into_conv2d(model.c14, model.b14)
            parameters["c14"] = {}
            parameters["c14"]["weight"] = conv14_weight
            parameters["c14"]["bias"] = conv14_bias

            ttnn_module_args.c15["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c15["use_shallow_conv_variant"] = False

            conv15_weight, conv15_bias = fold_batch_norm2d_into_conv2d(model.c15, model.b15)
            update_ttnn_module_args(ttnn_module_args.c15)
            parameters["c15"], c15_parallel_config = preprocess_conv2d(
                conv15_weight, conv15_bias, ttnn_module_args.c15, return_parallel_config=True
            )

            ttnn_module_args.c16["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c16["activation"] = "relu"

            conv16_weight, conv16_bias = fold_batch_norm2d_into_conv2d(model.c16, model.b16)
            update_ttnn_module_args(ttnn_module_args.c16)
            parameters["c16"], c16_parallel_config = preprocess_conv2d(
                conv16_weight, conv16_bias, ttnn_module_args.c16, return_parallel_config=True
            )

            conv17_weight, conv17_bias = fold_batch_norm2d_into_conv2d(model.c17, model.b17)
            parameters["c17"] = {}
            parameters["c17"]["weight"] = conv17_weight
            parameters["c17"]["bias"] = conv17_bias

            ttnn_module_args.c18["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c18["use_shallow_conv_variant"] = False

            conv18_weight, conv18_bias = fold_batch_norm2d_into_conv2d(model.c18, model.b18)
            update_ttnn_module_args(ttnn_module_args.c18)
            parameters["c18"], c18_parallel_config = preprocess_conv2d(
                conv18_weight, conv18_bias, ttnn_module_args.c18, return_parallel_config=True
            )

            ttnn_module_args.c19["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c19["activation"] = "relu"

            conv19_weight, conv19_bias = fold_batch_norm2d_into_conv2d(model.c19, model.b19)
            update_ttnn_module_args(ttnn_module_args.c19)
            parameters["c19"], c19_parallel_config = preprocess_conv2d(
                conv19_weight, conv19_bias, ttnn_module_args.c19, return_parallel_config=True
            )

            conv20_weight, conv20_bias = fold_batch_norm2d_into_conv2d(model.c20, model.b20)
            parameters["c20"] = {}
            parameters["c20"]["weight"] = conv20_weight
            parameters["c20"]["bias"] = conv20_bias

            ttnn_module_args.c21["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c21["use_shallow_conv_variant"] = False

            conv21_weight, conv21_bias = fold_batch_norm2d_into_conv2d(model.c21, model.b21)
            update_ttnn_module_args(ttnn_module_args.c21)
            parameters["c21"], c21_parallel_config = preprocess_conv2d(
                conv21_weight, conv21_bias, ttnn_module_args.c21, return_parallel_config=True
            )

            ttnn_module_args.c22["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c22["activation"] = "relu"

            conv22_weight, conv22_bias = fold_batch_norm2d_into_conv2d(model.c22, model.b22)
            update_ttnn_module_args(ttnn_module_args.c22)
            parameters["c22"], c22_parallel_config = preprocess_conv2d(
                conv22_weight, conv22_bias, ttnn_module_args.c22, return_parallel_config=True
            )

            conv23_weight, conv23_bias = fold_batch_norm2d_into_conv2d(model.c23, model.b23)
            parameters["c23"] = {}
            parameters["c23"]["weight"] = conv23_weight
            parameters["c23"]["bias"] = conv23_bias

            ttnn_module_args.c24["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c24["use_shallow_conv_variant"] = False

            conv24_weight, conv24_bias = fold_batch_norm2d_into_conv2d(model.c24, model.b24)
            update_ttnn_module_args(ttnn_module_args.c24)
            parameters["c24"], c24_parallel_config = preprocess_conv2d(
                conv24_weight, conv24_bias, ttnn_module_args.c24, return_parallel_config=True
            )

            ttnn_module_args.c25["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c25["activation"] = "relu"

            conv25_weight, conv25_bias = fold_batch_norm2d_into_conv2d(model.c25, model.b25)
            update_ttnn_module_args(ttnn_module_args.c25)
            parameters["c25"], c25_parallel_config = preprocess_conv2d(
                conv25_weight, conv25_bias, ttnn_module_args.c25, return_parallel_config=True
            )

            conv26_weight, conv26_bias = fold_batch_norm2d_into_conv2d(model.c26, model.b26)
            parameters["c26"] = {}
            parameters["c26"]["weight"] = conv26_weight
            parameters["c26"]["bias"] = conv26_bias

            ttnn_module_args.c27["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c27["use_shallow_conv_variant"] = False
            conv27_weight, conv27_bias = fold_batch_norm2d_into_conv2d(model.c27, model.b27)
            update_ttnn_module_args(ttnn_module_args.c27)
            parameters["c27"], c27_parallel_config = preprocess_conv2d(
                conv27_weight, conv27_bias, ttnn_module_args.c27, return_parallel_config=True
            )

            ttnn_module_args.c28["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c28["activation"] = "relu"

            conv28_weight, conv28_bias = fold_batch_norm2d_into_conv2d(model.c28, model.b28)
            update_ttnn_module_args(ttnn_module_args.c28)
            parameters["c28"], c28_parallel_config = preprocess_conv2d(
                conv28_weight, conv28_bias, ttnn_module_args.c28, return_parallel_config=True
            )

            conv29_weight, conv29_bias = fold_batch_norm2d_into_conv2d(model.c29, model.b29)
            parameters["c29"] = {}
            parameters["c29"]["weight"] = conv29_weight
            parameters["c29"]["bias"] = conv29_bias

            ttnn_module_args.c30["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c30["use_shallow_conv_variant"] = False

            conv30_weight, conv30_bias = fold_batch_norm2d_into_conv2d(model.c30, model.b30)
            update_ttnn_module_args(ttnn_module_args.c30)
            parameters["c30"], c30_parallel_config = preprocess_conv2d(
                conv30_weight, conv30_bias, ttnn_module_args.c30, return_parallel_config=True
            )

            ttnn_module_args.c31["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c31["activation"] = "relu"

            conv31_weight, conv31_bias = fold_batch_norm2d_into_conv2d(model.c31, model.b31)
            update_ttnn_module_args(ttnn_module_args.c31)
            parameters["c31"], c31_parallel_config = preprocess_conv2d(
                conv31_weight, conv31_bias, ttnn_module_args.c31, return_parallel_config=True
            )

            conv32_weight, conv32_bias = fold_batch_norm2d_into_conv2d(model.c32, model.b32)
            parameters["c32"] = {}
            parameters["c32"]["weight"] = conv32_weight
            parameters["c32"]["bias"] = conv32_bias

            ttnn_module_args.c33["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c33["use_shallow_conv_variant"] = False

            conv33_weight, conv33_bias = fold_batch_norm2d_into_conv2d(model.c33, model.b33)
            update_ttnn_module_args(ttnn_module_args.c33)
            parameters["c33"], c33_parallel_config = preprocess_conv2d(
                conv33_weight, conv33_bias, ttnn_module_args.c33, return_parallel_config=True
            )

            ttnn_module_args.c34["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c34["activation"] = "relu"

            conv34_weight, conv34_bias = fold_batch_norm2d_into_conv2d(model.c34, model.b34)
            update_ttnn_module_args(ttnn_module_args.c34)
            parameters["c34"], c34_parallel_config = preprocess_conv2d(
                conv34_weight, conv34_bias, ttnn_module_args.c34, return_parallel_config=True
            )

            conv35_weight, conv35_bias = fold_batch_norm2d_into_conv2d(model.c35, model.b35)
            parameters["c35"] = {}
            parameters["c35"]["weight"] = conv35_weight
            parameters["c35"]["bias"] = conv35_bias

            ttnn_module_args.c36["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c36["use_shallow_conv_variant"] = False

            conv36_weight, conv36_bias = fold_batch_norm2d_into_conv2d(model.c36, model.b36)
            update_ttnn_module_args(ttnn_module_args.c36)
            parameters["c36"], c36_parallel_config = preprocess_conv2d(
                conv36_weight, conv36_bias, ttnn_module_args.c36, return_parallel_config=True
            )

            ttnn_module_args.c37["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c37["activation"] = "relu"

            conv37_weight, conv37_bias = fold_batch_norm2d_into_conv2d(model.c37, model.b37)
            update_ttnn_module_args(ttnn_module_args.c37)
            parameters["c37"], c37_parallel_config = preprocess_conv2d(
                conv37_weight, conv37_bias, ttnn_module_args.c37, return_parallel_config=True
            )

            conv38_weight, conv38_bias = fold_batch_norm2d_into_conv2d(model.c38, model.b38)
            parameters["c38"] = {}
            parameters["c38"]["weight"] = conv38_weight
            parameters["c38"]["bias"] = conv38_bias

            ttnn_module_args.c39["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c39["use_shallow_conv_variant"] = False

            conv39_weight, conv39_bias = fold_batch_norm2d_into_conv2d(model.c39, model.b39)
            update_ttnn_module_args(ttnn_module_args.c39)
            parameters["c39"], c39_parallel_config = preprocess_conv2d(
                conv39_weight, conv39_bias, ttnn_module_args.c39, return_parallel_config=True
            )

            ttnn_module_args.c40["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c40["activation"] = "relu"

            conv40_weight, conv40_bias = fold_batch_norm2d_into_conv2d(model.c40, model.b40)
            update_ttnn_module_args(ttnn_module_args.c40)
            parameters["c40"], c40_parallel_config = preprocess_conv2d(
                conv40_weight, conv40_bias, ttnn_module_args.c40, return_parallel_config=True
            )

            conv41_weight, conv41_bias = fold_batch_norm2d_into_conv2d(model.c41, model.b41)
            parameters["c41"] = {}
            parameters["c41"]["weight"] = conv41_weight
            parameters["c41"]["bias"] = conv41_bias

            ttnn_module_args.c42["weights_dtype"] = ttnn.bfloat8_b

            conv42_weight, conv42_bias = fold_batch_norm2d_into_conv2d(model.c42, model.b42)
            update_ttnn_module_args(ttnn_module_args.c42)
            ttnn_module_args["c42"]["use_1d_systolic_array"] = True
            parameters["c42"], c42_parallel_config = preprocess_conv2d(
                conv42_weight, conv42_bias, ttnn_module_args.c42, return_parallel_config=True
            )

            ttnn_module_args.c43["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c43["activation"] = "relu"

            conv43_weight, conv43_bias = fold_batch_norm2d_into_conv2d(model.c43, model.b43)
            update_ttnn_module_args(ttnn_module_args.c43)
            parameters["c43"], c43_parallel_config = preprocess_conv2d(
                conv43_weight, conv43_bias, ttnn_module_args.c43, return_parallel_config=True
            )

            conv44_weight, conv44_bias = fold_batch_norm2d_into_conv2d(model.c44, model.b44)
            parameters["c44"] = {}
            parameters["c44"]["weight"] = conv44_weight
            parameters["c44"]["bias"] = conv44_bias

            ttnn_module_args.c45["weights_dtype"] = ttnn.bfloat8_b
            conv45_weight, conv45_bias = fold_batch_norm2d_into_conv2d(model.c45, model.b45)
            update_ttnn_module_args(ttnn_module_args.c45)
            ttnn_module_args["c45"]["use_1d_systolic_array"] = True
            parameters["c45"], c45_parallel_config = preprocess_conv2d(
                conv45_weight, conv45_bias, ttnn_module_args.c45, return_parallel_config=True
            )
            print(ttnn_module_args.c45)
            ttnn_module_args.c46["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c46["activation"] = "relu"

            conv46_weight, conv46_bias = fold_batch_norm2d_into_conv2d(model.c46, model.b46)
            update_ttnn_module_args(ttnn_module_args.c46)
            parameters["c46"], c46_parallel_config = preprocess_conv2d(
                conv46_weight, conv46_bias, ttnn_module_args.c46, return_parallel_config=True
            )

            conv47_weight, conv47_bias = fold_batch_norm2d_into_conv2d(model.c47, model.b47)
            parameters["c47"] = {}
            parameters["c47"]["weight"] = conv47_weight
            parameters["c47"]["bias"] = conv47_bias

            ttnn_module_args.c48["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c48["use_shallow_conv_variant"] = False

            conv48_weight, conv48_bias = fold_batch_norm2d_into_conv2d(model.c48, model.b48)
            update_ttnn_module_args(ttnn_module_args.c48)
            ttnn_module_args["c48"]["use_1d_systolic_array"] = True
            parameters["c48"], c48_parallel_config = preprocess_conv2d(
                conv48_weight, conv48_bias, ttnn_module_args.c48, return_parallel_config=True
            )

            ttnn_module_args.c49["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c49["activation"] = "relu"

            conv49_weight, conv49_bias = fold_batch_norm2d_into_conv2d(model.c49, model.b49)
            update_ttnn_module_args(ttnn_module_args.c49)
            parameters["c49"], c49_parallel_config = preprocess_conv2d(
                conv49_weight, conv49_bias, ttnn_module_args.c49, return_parallel_config=True
            )

            conv50_weight, conv50_bias = fold_batch_norm2d_into_conv2d(model.c50, model.b50)
            parameters["c50"] = {}
            parameters["c50"]["weight"] = conv50_weight
            parameters["c50"]["bias"] = conv50_bias

            ttnn_module_args.c51["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c51["use_shallow_conv_variant"] = False

            conv51_weight, conv51_bias = fold_batch_norm2d_into_conv2d(model.c51, model.b51)
            update_ttnn_module_args(ttnn_module_args.c51)
            ttnn_module_args["c51"]["use_1d_systolic_array"] = True
            parameters["c51"], c51_parallel_config = preprocess_conv2d(
                conv51_weight, conv51_bias, ttnn_module_args.c51, return_parallel_config=True
            )

            ttnn_module_args.c52["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c52["activation"] = "relu"

            conv52_weight, conv52_bias = fold_batch_norm2d_into_conv2d(model.c52, model.b52)
            update_ttnn_module_args(ttnn_module_args.c52)
            ttnn_module_args["c52"]["use_1d_systolic_array"] = True
            parameters["c52"], c52_parallel_config = preprocess_conv2d(
                conv52_weight, conv52_bias, ttnn_module_args.c52, return_parallel_config=True
            )
            parameters["l1"] = {}
            parameters["l1"]["weight"] = model.l1.weight
            parameters["l1"]["bias"] = model.l1.bias
            return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_wormhole_b0()
def test_mobilenetv2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/mobilenetv2/mobilenet_v2-b0353104.pth")
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("features.") or k.startswith("classifier."))}

    torch_model = Mobilenetv2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 input channels, 128x128 height and width
    torch_output_tensor = torch_model(torch_input_tensor)
    print("torch_output_tensor: ", torch_output_tensor.shape)
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtMobilenetv2(parameters)

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)
    print("output_tensor: ", output_tensor.shape)

    #
    # Tensor Postprocessing
    #
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(1, 1280)
    # output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
