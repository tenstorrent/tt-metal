# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn

from torchview import draw_graph

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0

from models.experimental.functional_unet.tt import ttnn_functional_unet_cbr1

import ttnn


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256


def custom_preprocessor(model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, CBR1):
        ttnn_module_args.c1["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c1_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c1["deallocate_activation"] = True
        ttnn_module_args.c1_2["deallocate_activation"] = True
        ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}
        ttnn_module_args.c1_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}

        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
        conv1_2_weight, conv1_2_bias = fold_batch_norm2d_into_conv2d(model.c1_2, model.b1_2)

        update_ttnn_module_args(ttnn_module_args.c1)
        update_ttnn_module_args(ttnn_module_args.c1_2)

        parameters["c1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.c1)
        parameters["c1_2"] = preprocess_conv2d(conv1_2_weight, conv1_2_bias, ttnn_module_args.c1_2)

    return parameters


class CBR1(nn.Module):
    def __init__(self):
        super(CBR1, self).__init__()
        # Contracting Path
        self.c1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.r1 = nn.ReLU(inplace=True)
        self.c1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.b1_2 = nn.BatchNorm2d(16)
        self.r1_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Contracting Path
        c1 = self.c1(x)
        b1 = self.b1(c1)
        r1 = self.r1(b1)
        c1_2 = self.c1_2(r1)
        b1_2 = self.b1_2(c1_2)
        r1_2 = self.r1_2(b1_2)

        output = r1_2
        return output


class CBR1_MAXPOOL(nn.Module):
    def __init__(self):
        super(CBR1_MAXPOOL, self).__init__()
        # Contracting Path
        self.c1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.r1 = nn.ReLU(inplace=True)
        self.c1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.b1_2 = nn.BatchNorm2d(16)
        self.r1_2 = nn.ReLU(inplace=True)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting Path
        c1 = self.c1(x)
        b1 = self.b1(c1)
        r1 = self.r1(b1)
        c1_2 = self.c1_2(r1)
        b1_2 = self.b1_2(c1_2)
        r1_2 = self.r1_2(b1_2)
        p1 = self.p1(r1_2)

        output = p1
        return output


@skip_for_wormhole_b0()
def test_cbr1():
    device_id = 0
    device = ttnn.open(device_id)

    torch.manual_seed(0)

    torch_model = CBR1()
    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    for name, parameter in torch_model.state_dict().items():
        if isinstance(parameter, torch.FloatTensor):
            new_state_dict[name] = parameter + 100.0

    torch_model.load_state_dict(new_state_dict)

    torch_input_tensor = torch.randn(2, 3, 1056, 160)  # Batch size of 2, 3 channels (RGB), 1056x160 input
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = ttnn_functional_unet_cbr1.CBR1(parameters)
    output_tensor = ttnn_model.torch_call(torch_input_tensor)
    print("the shape of output_tensor: ", output_tensor.size())
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9)
    ttnn.close(device)


@skip_for_wormhole_b0()
def test_cbr1_maxpool2d():
    device_id = 0
    device = ttnn.open(device_id)

    torch.manual_seed(0)

    torch_model = CBR1_MAXPOOL()
    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    for name, parameter in torch_model.state_dict().items():
        if isinstance(parameter, torch.FloatTensor):
            new_state_dict[name] = parameter  # + 100.0

    torch_model.load_state_dict(new_state_dict)

    torch_input_tensor = torch.randn(2, 3, 1056, 160)  # Batch size of 2, 3 channels (RGB), 1056x160 input
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: CBR1(),
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = ttnn_functional_unet_cbr1.CBR1(parameters)
    output_tensor = ttnn_model.torch_call(torch_input_tensor)
    output_tensor = torch.nn.functional.max_pool2d(output_tensor, kernel_size=2, stride=2)
    print("the shape of output_tensor: ", output_tensor.size())
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9)
    ttnn.close(device)
