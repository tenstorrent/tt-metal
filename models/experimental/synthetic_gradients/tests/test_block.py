# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torchvision import transforms, datasets

import ttnn
from models.utility_functions import tilize_to_list, untilize, comp_allclose_and_pcc

epsilon = 1e-5


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


def ttBatchnorm1d_inference(gamma, beta, running_mean, running_var, epsilon):
    def batchnorm1d_inference_(X):
        var_plus_eps = ttnn.add(epsilon, running_var)
        sqrt_var = ttnn.sqrt(var_plus_eps)
        sqrt_inv = ttnn.reciprocal(sqrt_var)
        x_minus_mean = ttnn.sub(X, running_mean)
        x_div_sqrt = ttnn.mul(x_minus_mean, sqrt_inv)
        x_gamma = ttnn.mul(x_div_sqrt, gamma)
        Y = ttnn.add(x_gamma, beta)
        return Y

    return batchnorm1d_inference_


class PytorchBatchNorm1D(nn.Module):
    def __init__(self, input_dim):
        super(PytorchBatchNorm1D, self).__init__()

        self.batchnorm1d_1 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        bn1_out = self.batchnorm1d_1(x)

        return bn1_out


def run_block_inference(in_features, out_features, device):
    # set inputs
    inputs_torch = torch.FloatTensor(1, in_features).uniform_(-1.0, 1.0).requires_grad_(True)

    inputs_reshape = inputs_torch.reshape(1, 1, 1, -1)
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

    # torch linear params
    weight_lin_torch = torch.randn(out_features, in_features)
    bias_lin_torch = torch.randn(out_features)
    linear_torch = torchLinear(in_features, out_features, weight_lin_torch, bias_lin_torch)

    # tt linear params
    weight_lin = weight_lin_torch.view(1, 1, out_features, in_features)
    tilized_weight_lin_tt = tilize_to_list(weight_lin)
    weight_lin_tt = ttnn.Tensor(
        tilized_weight_lin_tt,
        weight_lin.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    bias_lin_src = bias_lin_torch.view(1, 1, 1, out_features)
    bias_lin = torch.zeros(1, 1, 32, out_features)
    bias_lin[:, :, :1, :] = bias_lin_src
    tilized_bias_lin_tt = tilize_to_list(bias_lin)
    bias_lin_tt = ttnn.Tensor(
        tilized_bias_lin_tt,
        bias_lin.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    # batch norm torch
    bn_torch = PytorchBatchNorm1D(out_features)
    bn_torch.eval()
    weight_bn_torch = torch.nn.Parameter(torch.FloatTensor(out_features).uniform_(-1.0, 1.0).requires_grad_(True))
    bias_bn_torch = torch.nn.Parameter(torch.FloatTensor(out_features).uniform_(-1.0, 1.0).requires_grad_(True))
    running_mean_bn_torch = torch.FloatTensor(out_features).uniform_(-1.0, 1.0).requires_grad_(False)
    running_var_bn_torch = torch.FloatTensor(out_features).uniform_(0.0, 1.0).requires_grad_(False)  # must be positive

    bn_torch.batchnorm1d_1.weight = weight_bn_torch
    bn_torch.batchnorm1d_1.bias = bias_bn_torch
    bn_torch.batchnorm1d_1.running_mean = running_mean_bn_torch
    bn_torch.batchnorm1d_1.running_var = running_var_bn_torch
    bn_torch.batchnorm1d_1.eps = epsilon

    # batch norm tt
    weight_bn_src = weight_bn_torch.view(1, 1, 1, out_features)
    weight_bn_tt = torch.zeros(1, 1, 32, out_features)
    weight_bn_tt[:, :, :1, :] = weight_bn_src
    tilized_weight_bn_tt = tilize_to_list(weight_bn_tt)
    gamma = ttnn.Tensor(
        tilized_weight_bn_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    bias_bn_src = bias_bn_torch.view(1, 1, 1, out_features)
    bias_bn_tt = torch.zeros(1, 1, 32, out_features)
    bias_bn_tt[:, :, :1, :] = bias_bn_src
    tilized_bias_bn_tt = tilize_to_list(bias_bn_tt)
    beta = ttnn.Tensor(
        tilized_bias_bn_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_mean_bn_src = running_mean_bn_torch.view(1, 1, 1, out_features)
    running_mean_bn_tt = torch.zeros(1, 1, 32, out_features)
    running_mean_bn_tt[:, :, :1, :] = running_mean_bn_src
    tilized_running_mean_tt = tilize_to_list(running_mean_bn_tt)
    running_mean_tt = ttnn.Tensor(
        tilized_running_mean_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_var_bn_src = running_var_bn_torch.view(1, 1, 1, out_features)
    running_var_bn_tt = torch.zeros(1, 1, 32, out_features)
    running_var_bn_tt[:, :, :1, :] = running_var_bn_src
    tilized_running_var_tt = tilize_to_list(running_var_bn_tt)
    running_var_tt = ttnn.Tensor(
        tilized_running_var_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    epsilon_torch = torch.tensor([[[out_features * [epsilon]]]])
    epsilon_tor = torch.zeros(1, 1, 32, out_features)
    epsilon_tor[:, :, :1, :] = epsilon_torch
    tilized_eps_tt = tilize_to_list(epsilon_tor)
    eps_tt = ttnn.Tensor(
        tilized_eps_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    # run through the models
    # torch
    output_lin_torch = linear_torch(inputs_torch)
    output_bn_torch = bn_torch(output_lin_torch)
    output_full_torch = torch.nn.functional.relu(output_bn_torch)

    # tt
    linear_tt = ttLinear(weight_lin_tt, bias_lin_tt)
    output_lin_tt = linear_tt(inputs_tt)
    bn_tt = ttBatchnorm1d_inference(gamma, beta, running_mean_tt, running_var_tt, eps_tt)
    output_bn_tt = bn_tt(output_lin_tt)
    output_full_tt = ttnn.relu(output_bn_tt)

    # compare
    output_lin_tt_untilized = untilize(torch.Tensor(output_lin_tt.cpu().to_torch()))
    output_lin_tt_untilized = output_lin_tt_untilized[0, 0, 0, :]

    output_bn_tt_untilized = untilize(torch.Tensor(output_bn_tt.cpu().to_torch()))
    output_bn_tt_untilized = output_bn_tt_untilized[0, 0, 0, :]

    output_full_tt_untilized = untilize(torch.Tensor(output_full_tt.cpu().to_torch()))
    output_full_tt_untilized = output_full_tt_untilized[0, 0, 0, :]

    print("pytorch_linear_out:", output_lin_torch[0][0:10])
    print("tt_linear_out:", output_lin_tt_untilized[0:10])

    liner_test_result, output = comp_allclose_and_pcc(output_lin_torch[0], output_lin_tt_untilized)
    print("\n\n", "atol/rtol:", liner_test_result, "| output:", output, "\n\n")

    print("pytorch_bn_out:", output_bn_torch[0][0:10])
    print("tt_bn_out:", output_bn_tt_untilized[0:10])

    bn_test_result, output = comp_allclose_and_pcc(output_bn_torch[0], output_bn_tt_untilized)
    print("\n\n", "atol/rtol:", bn_test_result, "| output:", output, "\n\n")

    print("pytorch_full_out:", output_full_torch[0][0:10])
    print("tt_full_out:", output_full_tt_untilized[0:10])

    full_test_result, output = comp_allclose_and_pcc(output_full_torch[0], output_full_tt_untilized)
    print("\n\n", "atol/rtol:", full_test_result, "| output:", output, "\n\n")


def test_run_block_inference(device):
    run_block_inference(1024, 256, device)
