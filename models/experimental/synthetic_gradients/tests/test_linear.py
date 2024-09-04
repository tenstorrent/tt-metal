# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torchvision import transforms, datasets

import ttnn
from models.utility_functions import tilize_to_list, untilize, comp_allclose_and_pcc


def ttLinear(weight, bias):
    def linear_(activation):
        weight_T = ttnn.transpose(weight, -2, -1)
        output = ttnn.matmul(activation, weight_T)
        output_plus_bias = ttnn.add(output, bias)
        return output_plus_bias

    return linear_


def torchLinear(in_features, out_features, weight, bias):
    linear_torch = torch.nn.Linear(out_features, in_features)
    linear_torch.weight = nn.Parameter(weight)
    linear_torch.bias = nn.Parameter(bias)

    return linear_torch


def run_linear_test(in_features, out_features, device):
    # torch
    torch_input_tensor = torch.randn(1, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    linear_torch = torchLinear(in_features, out_features, weight, bias)
    output_torch = linear_torch(torch_input_tensor)

    # tt
    weight_tt = weight.view(1, 1, out_features, in_features)
    bias_src = bias.view(1, 1, 1, out_features)
    bias_tt = torch.zeros(1, 1, 32, out_features)
    bias_tt[:, :, :1, :] = bias_src

    inputs_reshape = torch_input_tensor.reshape(1, 1, 1, -1)
    inputs_targ = torch.zeros(1, 1, 32, inputs_reshape.shape[3])
    inputs_targ[:, :, :1, :] = inputs_reshape
    tilized_inputs = tilize_to_list(inputs_targ)
    inputs_tt = ttnn.Tensor(
        tilized_inputs,
        inputs_targ.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    weight_tt = tilize_to_list(weight_tt)
    bias_tt = tilize_to_list(bias_tt)
    weight_tt = ttnn.Tensor(
        weight_tt,
        [1, 1, out_features, in_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )
    bias_tt = ttnn.Tensor(
        bias_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    linear_tt = ttLinear(weight_tt, bias_tt)
    output_tt = linear_tt(inputs_tt)
    output_tt = untilize(torch.Tensor(output_tt.cpu().to_torch()))
    output_tt = output_tt[0, 0, 0, :]

    test_results, output = comp_allclose_and_pcc(output_torch, output_tt)

    print("\n\n", "atol/rtol:", test_results, "| output:", output, "\n\n")


def test_linear_test(device):
    run_linear_test(1024, 256, device)
