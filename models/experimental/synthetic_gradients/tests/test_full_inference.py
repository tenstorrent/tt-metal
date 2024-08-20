# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torchvision import transforms, datasets

import ttnn

from models.utility_functions import tilize_to_list, untilize, comp_allclose_and_pcc

epsilon1 = 1e-5
epsilon2 = 1e-5


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


def run_full_inference(in_features, hidden_features, out_features, device):
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

    #### Layer 1
    # torch linear params layer1
    weight1_lin_torch = torch.randn(hidden_features, in_features)
    bias1_lin_torch = torch.randn(hidden_features)
    linear1_torch = torchLinear(in_features, hidden_features, weight1_lin_torch, bias1_lin_torch)

    # tt linear params layer1
    weight1_lin = weight1_lin_torch.view(1, 1, hidden_features, in_features)
    tilized_weight1_lin_tt = tilize_to_list(weight1_lin)
    weight1_lin_tt = ttnn.Tensor(
        tilized_weight1_lin_tt,
        weight1_lin.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    bias1_lin_src = bias1_lin_torch.view(1, 1, 1, hidden_features)
    bias1_lin = torch.zeros(1, 1, 32, hidden_features)
    bias1_lin[:, :, :1, :] = bias1_lin_src
    tilized_bias1_lin_tt = tilize_to_list(bias1_lin)
    bias1_lin_tt = ttnn.Tensor(
        tilized_bias1_lin_tt,
        bias1_lin.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    # batch norm torch layer1
    bn1_torch = PytorchBatchNorm1D(hidden_features)
    bn1_torch.eval()
    weight1_bn_torch = torch.nn.Parameter(torch.FloatTensor(hidden_features).uniform_(-1.0, 1.0).requires_grad_(True))
    bias1_bn_torch = torch.nn.Parameter(torch.FloatTensor(hidden_features).uniform_(-1.0, 1.0).requires_grad_(True))
    running_mean1_bn_torch = torch.FloatTensor(hidden_features).uniform_(-1.0, 1.0).requires_grad_(False)
    running_var1_bn_torch = (
        torch.FloatTensor(hidden_features).uniform_(0.0, 1.0).requires_grad_(False)
    )  # must be positive

    bn1_torch.batchnorm1d_1.weight = weight1_bn_torch
    bn1_torch.batchnorm1d_1.bias = bias1_bn_torch
    bn1_torch.batchnorm1d_1.running_mean = running_mean1_bn_torch
    bn1_torch.batchnorm1d_1.running_var = running_var1_bn_torch
    bn1_torch.batchnorm1d_1.eps = epsilon1

    # batch norm tt layer 1
    weight1_bn_src = weight1_bn_torch.view(1, 1, 1, hidden_features)
    weight1_bn_tt = torch.zeros(1, 1, 32, hidden_features)
    weight1_bn_tt[:, :, :1, :] = weight1_bn_src
    tilized_weight1_bn_tt = tilize_to_list(weight1_bn_tt)
    gamma1 = ttnn.Tensor(
        tilized_weight1_bn_tt,
        [1, 1, 32, hidden_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    bias1_bn_src = bias1_bn_torch.view(1, 1, 1, hidden_features)
    bias1_bn_tt = torch.zeros(1, 1, 32, hidden_features)
    bias1_bn_tt[:, :, :1, :] = bias1_bn_src
    tilized_bias1_bn_tt = tilize_to_list(bias1_bn_tt)
    beta1 = ttnn.Tensor(
        tilized_bias1_bn_tt,
        [1, 1, 32, hidden_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_mean1_bn_src = running_mean1_bn_torch.view(1, 1, 1, hidden_features)
    running_mean1_bn_tt = torch.zeros(1, 1, 32, hidden_features)
    running_mean1_bn_tt[:, :, :1, :] = running_mean1_bn_src
    tilized_running_mean1_tt = tilize_to_list(running_mean1_bn_tt)
    running_mean1_tt = ttnn.Tensor(
        tilized_running_mean1_tt,
        [1, 1, 32, hidden_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_var1_bn_src = running_var1_bn_torch.view(1, 1, 1, hidden_features)
    running_var1_bn_tt = torch.zeros(1, 1, 32, hidden_features)
    running_var1_bn_tt[:, :, :1, :] = running_var1_bn_src
    tilized_running_var1_tt = tilize_to_list(running_var1_bn_tt)
    running_var1_tt = ttnn.Tensor(
        tilized_running_var1_tt,
        [1, 1, 32, hidden_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    epsilon1_torch = torch.tensor([[[hidden_features * [epsilon1]]]])
    epsilon1_tor = torch.zeros(1, 1, 32, hidden_features)
    epsilon1_tor[:, :, :1, :] = epsilon1_torch
    tilized_eps1_tt = tilize_to_list(epsilon1_tor)
    eps1_tt = ttnn.Tensor(
        tilized_eps1_tt,
        [1, 1, 32, hidden_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    ### Layer 2
    # torch linear params layer2
    weight2_lin_torch = torch.randn(out_features, hidden_features)
    bias2_lin_torch = torch.randn(out_features)
    linear2_torch = torchLinear(hidden_features, out_features, weight2_lin_torch, bias2_lin_torch)

    # tt linear params layer2
    weight2_lin = weight2_lin_torch.view(1, 1, out_features, hidden_features)
    tilized_weight2_lin_tt = tilize_to_list(weight2_lin)
    weight2_lin_tt = ttnn.Tensor(
        tilized_weight2_lin_tt,
        weight2_lin.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    bias2_lin_src = bias2_lin_torch.view(1, 1, 1, out_features)
    bias2_lin = torch.zeros(1, 1, 32, out_features)
    bias2_lin[:, :, :1, :] = bias2_lin_src
    tilized_bias2_lin_tt = tilize_to_list(bias2_lin)
    bias2_lin_tt = ttnn.Tensor(
        tilized_bias2_lin_tt,
        bias2_lin.shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    # batch norm torch layer2
    bn2_torch = PytorchBatchNorm1D(out_features)
    bn2_torch.eval()
    weight2_bn_torch = torch.nn.Parameter(torch.FloatTensor(out_features).uniform_(-1.0, 1.0).requires_grad_(True))
    bias2_bn_torch = torch.nn.Parameter(torch.FloatTensor(out_features).uniform_(-1.0, 1.0).requires_grad_(True))
    running_mean2_bn_torch = torch.FloatTensor(out_features).uniform_(-1.0, 1.0).requires_grad_(False)
    running_var2_bn_torch = torch.FloatTensor(out_features).uniform_(0.0, 1.0).requires_grad_(False)  # must be positive

    bn2_torch.batchnorm1d_1.weight = weight2_bn_torch
    bn2_torch.batchnorm1d_1.bias = bias2_bn_torch
    bn2_torch.batchnorm1d_1.running_mean = running_mean2_bn_torch
    bn2_torch.batchnorm1d_1.running_var = running_var2_bn_torch
    bn2_torch.batchnorm1d_1.eps = epsilon2

    # batch norm tt layer 2
    weight2_bn_src = weight2_bn_torch.view(1, 1, 1, out_features)
    weight2_bn_tt = torch.zeros(1, 1, 32, out_features)
    weight2_bn_tt[:, :, :1, :] = weight2_bn_src
    tilized_weight2_bn_tt = tilize_to_list(weight2_bn_tt)
    gamma2 = ttnn.Tensor(
        tilized_weight2_bn_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    bias2_bn_src = bias2_bn_torch.view(1, 1, 1, out_features)
    bias2_bn_tt = torch.zeros(1, 1, 32, out_features)
    bias2_bn_tt[:, :, :1, :] = bias2_bn_src
    tilized_bias2_bn_tt = tilize_to_list(bias2_bn_tt)
    beta2 = ttnn.Tensor(
        tilized_bias2_bn_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_mean2_bn_src = running_mean2_bn_torch.view(1, 1, 1, out_features)
    running_mean2_bn_tt = torch.zeros(1, 1, 32, out_features)
    running_mean2_bn_tt[:, :, :1, :] = running_mean2_bn_src
    tilized_running_mean2_tt = tilize_to_list(running_mean2_bn_tt)
    running_mean2_tt = ttnn.Tensor(
        tilized_running_mean2_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_var2_bn_src = running_var2_bn_torch.view(1, 1, 1, out_features)
    running_var2_bn_tt = torch.zeros(1, 1, 32, out_features)
    running_var2_bn_tt[:, :, :1, :] = running_var2_bn_src
    tilized_running_var2_tt = tilize_to_list(running_var2_bn_tt)
    running_var2_tt = ttnn.Tensor(
        tilized_running_var2_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    epsilon2_torch = torch.tensor([[[out_features * [epsilon2]]]])
    epsilon2_tor = torch.zeros(1, 1, 32, out_features)
    epsilon2_tor[:, :, :1, :] = epsilon2_torch
    tilized_eps2_tt = tilize_to_list(epsilon2_tor)
    eps2_tt = ttnn.Tensor(
        tilized_eps2_tt,
        [1, 1, 32, out_features],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    # run through the models
    # torch layer 1
    output_lin1_torch = linear1_torch(inputs_torch)
    output_bn1_torch = bn1_torch(output_lin1_torch)
    output_layer1_torch = torch.nn.functional.relu(output_bn1_torch)
    # torch layer 2
    output_lin2_torch = linear2_torch(output_layer1_torch)
    output_bn2_torch = bn2_torch(output_lin2_torch)
    output_layer2_torch = torch.nn.functional.relu(output_bn2_torch)

    # tt layer 1
    linear1_tt = ttLinear(weight1_lin_tt, bias1_lin_tt)
    output_lin1_tt = linear1_tt(inputs_tt)
    bn1_tt = ttBatchnorm1d_inference(gamma1, beta1, running_mean1_tt, running_var1_tt, eps1_tt)
    output_bn1_tt = bn1_tt(output_lin1_tt)
    output_layer1_tt = ttnn.relu(output_bn1_tt)
    # tt layer 2
    linear2_tt = ttLinear(weight2_lin_tt, bias2_lin_tt)
    output_lin2_tt = linear2_tt(output_layer1_tt)
    bn2_tt = ttBatchnorm1d_inference(gamma2, beta2, running_mean2_tt, running_var2_tt, eps2_tt)
    output_bn2_tt = bn2_tt(output_lin2_tt)
    output_layer2_tt = ttnn.relu(output_bn2_tt)

    # compare
    output_layer1_tt_untilized = untilize(torch.Tensor(output_layer1_tt.cpu().to_torch()))
    output_layer1_tt_untilized = output_layer1_tt_untilized[0, 0, 0, :]

    output_layer2_tt_untilized = untilize(torch.Tensor(output_layer2_tt.cpu().to_torch()))
    output_layer2_tt_untilized = output_layer2_tt_untilized[0, 0, 0, :]

    print("pytorch_layer1_out:", output_layer1_torch[0][0:10])
    print("tt_layer1_out:", output_layer1_tt_untilized[0:10])

    layer1_test_result, output = comp_allclose_and_pcc(output_layer1_torch[0], output_layer1_tt_untilized)
    print("\n\n", "atol/rtol 1:", layer1_test_result, "| output:", output, "\n\n")

    print("pytorch_layer2_out:", output_layer2_torch[0][0:10])
    print("tt_layer2_out:", output_layer2_tt_untilized[0:10])

    layer2_test_result, output = comp_allclose_and_pcc(output_layer2_torch[0], output_layer2_tt_untilized)
    print("\n\n", "atol/rtol 2:", layer2_test_result, "| output:", output, "\n\n")


def test_run_full_inference(device):
    run_full_inference(1024, 256, 32, device)
