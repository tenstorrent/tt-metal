# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torchvision import transforms, datasets

import ttnn
from models.utility_functions import tilize_to_list, untilize, comp_allclose_and_pcc

epsilon = 1e-5


def batchnorm1d_inference(gamma, beta, running_mean, running_var, epsilon):
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


def run_btchnorm_inference(bn_size, device):
    inputs = torch.FloatTensor(1, bn_size).uniform_(-1.0, 1.0).requires_grad_(True)
    # torch
    bn_torch = PytorchBatchNorm1D(bn_size)
    bn_torch.eval()
    weight_bn = torch.nn.Parameter(torch.FloatTensor(bn_size).uniform_(-1.0, 1.0).requires_grad_(True))
    bias_bn = torch.nn.Parameter(torch.FloatTensor(bn_size).uniform_(-1.0, 1.0).requires_grad_(True))
    running_mean = torch.FloatTensor(bn_size).uniform_(-1.0, 1.0).requires_grad_(False)
    running_var = torch.FloatTensor(bn_size).uniform_(0.0, 1.0).requires_grad_(False)  # must be positive

    bn_torch.batchnorm1d_1.weight = weight_bn
    bn_torch.batchnorm1d_1.bias = bias_bn
    bn_torch.batchnorm1d_1.running_mean = running_mean
    bn_torch.batchnorm1d_1.running_var = running_var
    bn_torch.batchnorm1d_1.eps = epsilon

    # tt
    weight_bn_src = weight_bn.view(1, 1, 1, bn_size)
    weight_bn_tt = torch.zeros(1, 1, 32, bn_size)
    weight_bn_tt[:, :, :1, :] = weight_bn_src
    tilized_weight_bn_tt = tilize_to_list(weight_bn_tt)
    gamma = ttnn.Tensor(
        tilized_weight_bn_tt,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    bias_bn_src = bias_bn.view(1, 1, 1, bn_size)
    bias_bn_tt = torch.zeros(1, 1, 32, bn_size)
    bias_bn_tt[:, :, :1, :] = bias_bn_src
    tilized_bias_bn_tt = tilize_to_list(bias_bn_tt)
    beta = ttnn.Tensor(
        tilized_bias_bn_tt,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_mean_bn_src = running_mean.view(1, 1, 1, bn_size)
    running_mean_bn_tt = torch.zeros(1, 1, 32, bn_size)
    running_mean_bn_tt[:, :, :1, :] = running_mean_bn_src
    tilized_running_mean_tt = tilize_to_list(running_mean_bn_tt)
    running_mean_tt = ttnn.Tensor(
        tilized_running_mean_tt,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_var_bn_src = running_var.view(1, 1, 1, bn_size)
    running_var_bn_tt = torch.zeros(1, 1, 32, bn_size)
    running_var_bn_tt[:, :, :1, :] = running_var_bn_src
    tilized_running_var_tt = tilize_to_list(running_var_bn_tt)
    running_var_tt = ttnn.Tensor(
        tilized_running_var_tt,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    epsilon_torch = torch.tensor([[[bn_size * [epsilon]]]])
    epsilon_tor = torch.zeros(1, 1, 32, bn_size)
    epsilon_tor[:, :, :1, :] = epsilon_torch
    tilized_eps_tt = tilize_to_list(epsilon_tor)
    eps_tt = ttnn.Tensor(
        tilized_eps_tt,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    inputs_bn_src = inputs.view(1, 1, 1, bn_size)
    inputs_bn_tt = torch.zeros(1, 1, 32, bn_size)
    inputs_bn_tt[:, :, :1, :] = inputs_bn_src
    tilized_inputs_tt = tilize_to_list(inputs_bn_tt)
    X_tt = ttnn.Tensor(
        tilized_inputs_tt,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    # run through models
    output_bn_torch = bn_torch(inputs)
    bn_tt = batchnorm1d_inference(gamma, beta, running_mean_tt, running_var_tt, eps_tt)
    output_bn_tt = bn_tt(X_tt)

    output_bn_tt_untilized = untilize(torch.Tensor(output_bn_tt.cpu().to_torch()))
    output_bn_tt_untilized = output_bn_tt_untilized[0, 0, 0, :]

    print("pytorch_out:", output_bn_torch[0][0:10])
    print("tt_out:", output_bn_tt_untilized[0:10])

    test_results, output = comp_allclose_and_pcc(output_bn_torch[0], output_bn_tt_untilized)

    print("\n\n", "atol/rtol:", test_results, "| pcc:", output, "\n\n")


def test_batchnorm_inference(device):
    run_btchnorm_inference(1024, device)
