# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_yolov4.tt import ttnn_yolov4

import time
import tt_lib as ttl
import tt_lib.profiler as profiler

import ttnn


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, Conv_Bn_Activation):
            ttnn_module_args.c1["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c1["use_shallow_conv_variant"] = (
                False if device.arch() == ttl.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c1["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c1["deallocate_activation"] = True
            ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = None

            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
            update_ttnn_module_args(ttnn_module_args.c1)
            parameters["c1"], c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
            )

            ttnn_module_args.c2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c2["use_shallow_conv_variant"] = (
                False if device.arch() == ttl.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2["activation"] = "relu"  # Fuse relu with conv2
            ttnn_module_args.c2["deallocate_activation"] = True
            ttnn_module_args.c2["conv_blocking_and_parallelization_config_override"] = None

            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
            update_ttnn_module_args(ttnn_module_args.c2)
            parameters["c2"], c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
            )

            ttnn_module_args.c3["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c3["use_shallow_conv_variant"] = (
                False if device.arch() == ttl.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c3["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c3["deallocate_activation"] = True
            ttnn_module_args.c3["conv_blocking_and_parallelization_config_override"] = None

            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
            update_ttnn_module_args(ttnn_module_args.c3)
            parameters["c3"], c3_parallel_config = preprocess_conv2d(
                conv3_weight, conv3_bias, ttnn_module_args.c3, return_parallel_config=True
            )

            ttnn_module_args.c4["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c4["use_shallow_conv_variant"] = (
                False if device.arch() == ttl.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c4["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c4["deallocate_activation"] = True
            ttnn_module_args.c4["conv_blocking_and_parallelization_config_override"] = None

            conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
            update_ttnn_module_args(ttnn_module_args.c4)
            parameters["c4"], c4_parallel_config = preprocess_conv2d(
                conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
            )

            ttnn_module_args.c5["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c5["use_shallow_conv_variant"] = (
                False if device.arch() == ttl.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c5["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c5["deallocate_activation"] = True
            ttnn_module_args.c5["conv_blocking_and_parallelization_config_override"] = None

            conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
            update_ttnn_module_args(ttnn_module_args.c5)
            parameters["c5"], c5_parallel_config = preprocess_conv2d(
                conv5_weight, conv5_bias, ttnn_module_args.c5, return_parallel_config=True
            )

            ttnn_module_args.c6["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c6["use_shallow_conv_variant"] = (
                False if device.arch() == ttl.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c6["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c6["deallocate_activation"] = True
            ttnn_module_args.c6["conv_blocking_and_parallelization_config_override"] = None

            conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
            update_ttnn_module_args(ttnn_module_args.c6)
            parameters["c6"], c6_parallel_config = preprocess_conv2d(
                conv6_weight, conv6_bias, ttnn_module_args.c6, return_parallel_config=True
            )

            ttnn_module_args.c7["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c7["use_shallow_conv_variant"] = (
                False if device.arch() == ttl.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c7["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c7["deallocate_activation"] = True
            ttnn_module_args.c7["conv_blocking_and_parallelization_config_override"] = None

            conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
            update_ttnn_module_args(ttnn_module_args.c7)
            parameters["c7"], c7_parallel_config = preprocess_conv2d(
                conv7_weight, conv7_bias, ttnn_module_args.c7, return_parallel_config=True
            )

            ttnn_module_args.c8["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c8["use_shallow_conv_variant"] = (
                False if device.arch() == ttl.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c8["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c8["deallocate_activation"] = True
            ttnn_module_args.c8["conv_blocking_and_parallelization_config_override"] = None

            conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
            update_ttnn_module_args(ttnn_module_args.c8)
            parameters["c8"], c8_parallel_config = preprocess_conv2d(
                conv8_weight, conv8_bias, ttnn_module_args.c8, return_parallel_config=True
            )

        return parameters

    return custom_preprocessor


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bn=True, bias=True):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
        self.b1 = nn.BatchNorm2d(out_channels)
        # self.mish = Mish()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        # x = self.mish
        x = self.relu(x)
        return x


######################


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.b1 = nn.BatchNorm2d(32)
        # self.mish = Mish()
        self.relu = nn.ReLU(inplace=True)

        # self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.c2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.b2 = nn.BatchNorm2d(64)
        # self.mish = Mish()
        # self.relu = nn.ReLU(inplace=True)

        # self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.c3 = nn.Conv2d(64, 64, 1, 1, 0)
        self.b3 = nn.BatchNorm2d(64)
        #        # [route]
        #        # layers = -2
        #        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.c4 = nn.Conv2d(64, 64, 1, 1, 0)
        self.b4 = nn.BatchNorm2d(64)
        #
        #        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.c5 = nn.Conv2d(64, 32, 1, 1, 0)
        self.b5 = nn.BatchNorm2d(32)
        #        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        self.c6 = nn.Conv2d(32, 64, 3, 1, 1)
        self.b6 = nn.BatchNorm2d(64)
        #        # [shortcut]
        #        # from=-3
        #        # activation = linear
        #
        #        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.c7 = nn.Conv2d(64, 64, 1, 1, 0)
        self.b7 = nn.BatchNorm2d(64)
        #        # [route]
        #        # layers = -1, -7
        #        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        self.c8 = nn.Conv2d(128, 64, 1, 1, 0)
        self.b8 = nn.BatchNorm2d(64)

    def forward(self, input):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_r = self.relu(x1_b)
        x2 = self.c2(x1_r)
        x2_b = self.b2(x2)
        x2_r = self.relu(x2_b)
        x3 = self.c3(x2_r)
        x3_b = self.b3(x3)
        x3_r = self.relu(x3_b)
        x4 = self.c4(x3_r)
        x4_b = self.b4(x4)
        x4_r = self.relu(x4_b)
        x5 = self.c5(x4_r)
        x5_b = self.b5(x5)
        x5_r = self.relu(x5_b)
        x6 = self.c6(x5_r)
        x6_b = self.b6(x6)
        x6_r = self.relu(x6_b)
        #        # route -2
        #        x4 = self.conv4(x2)
        #        x5 = self.conv5(x4)
        #        x6 = self.conv6(x5)
        #        # shortcut -3
        x6_r = x6_r + x4_r
        #
        #        x7 = self.conv7(x6)
        x7 = self.c7(x6_r)
        x7_b = self.b7(x7)
        x7_r = self.relu(x7_b)
        #        # [route]
        #        # layers = -1, -7
        x7_r = torch.cat([x7_r, x3_r], dim=1)
        x8 = self.c8(x7_r)
        x8_b = self.b8(x8)
        x8_r = self.relu(x8_b)
        #        x8 = self.conv8(x7)
        #        return x8
        return x8_r


######################


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", default=0, type=int)
    args = ap.parse_args()

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    torch.manual_seed(0)

    torch_model = DownSample1()
    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    for name, parameter in torch_model.state_dict().items():
        if isinstance(parameter, torch.FloatTensor):
            new_state_dict[name] = parameter + 100.0

    torch_model.load_state_dict(new_state_dict)

    torch_input_tensor = torch.randn(1, 3, 320, 320)  # Batch size of 1, 128 input channels, 160x160 height and width
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = ttnn_yolov4.Yolov4(parameters)

    # Tensor Preprocessing
    #
    input_shape = torch_input_tensor.shape
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(1, 160, 160, 64)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.8)
    ttnn.close_device(device)
