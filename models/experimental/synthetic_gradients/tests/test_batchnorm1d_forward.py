# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torchvision import transforms, datasets

import ttnn
from models.utility_functions import (
    tilize_to_list,
    untilize,
    comp_allclose_and_pcc,
    comp_pcc,
)

from tt_lib.utils import pad_activation, pad_weight


# tt_metal
def tt_batch_norm(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps: float,
    momentum: float,
    bn_size,
    mode="train",
    device=None,
):
    H = 32
    W = bn_size
    batch_size = x.shape.with_tile_padding()[0]
    print("batch_size:", batch_size)
    epsilon_torch = torch.tensor([[[W * [eps]]]])
    epsilon_padded = pad_activation(epsilon_torch)
    epsilon_tilized = tilize_to_list(epsilon_padded)
    eps_tt = ttnn.Tensor(
        epsilon_tilized,
        [1, 1, H, W],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    # Use is_grad_enabled to determine whether we are in training mode
    if mode == "inference":
        print("inference mode")
        assert batch_size == 1, "in inference mode batch size must be 1!"
        # In prediction mode, use mean and variance obtained by moving average
        var_plus_eps = ttnn.add(eps_tt, running_var)
        sqrt_var = ttnn.sqrt(var_plus_eps)
        sqrt_inv = ttnn.reciprocal(sqrt_var)
        x_minus_mean = ttnn.sub(x, running_mean)
        x_div_sqrt = ttnn.mul(x_minus_mean, sqrt_inv)
        x_gamma = ttnn.mul(x_div_sqrt, gamma)
        Y = ttnn.add(x_gamma, beta)
    else:
        print("train mode")
        x_tor = x.cpu().to_torch()
        x_tor = x_tor.view(batch_size, 1, H, W)
        x_tor = untilize(x_tor)
        mean = x_tor.mean(dim=0)
        var = ((x_tor - mean) ** 2).mean(dim=0)
        var_reshaped = var.view(1, 1, 32, 32)
        var_tilized = tilize_to_list(var_reshaped)
        var_tt = ttnn.Tensor(
            var_tilized,
            [1, 1, H, W],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )

        # In training mode, the current mean and variance are used
        var_plus_eps = ttnn.add(eps_tt, var_tt)
        sqrt_var = ttnn.sqrt(var_plus_eps)
        sqrt_inv = ttnn.reciprocal(sqrt_var)
        sqrt_inv_data = sqrt_inv.cpu().to_torch()
        sqrt_inv_data = torch.Tensor(sqrt_inv_data).reshape((1, 1, H, W))
        sqrt_inv_data = untilize(sqrt_inv_data)
        x_minus_mean = x_tor - mean
        x_div_sqrt_tor = torch.mul(x_minus_mean, sqrt_inv_data)

        # Update the mean and variance using moving average
        momentum_torch = torch.tensor([[[W * [momentum]]]])
        momentum_padded = pad_activation(momentum_torch)
        momentum_tilized = tilize_to_list(momentum_padded)
        momentum_tt = ttnn.Tensor(
            momentum_tilized,
            [1, 1, H, W],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )
        ones_torch = torch.tensor([[[bn_size * [1.0]]]])
        ones_padded = pad_activation(ones_torch)
        ones_tilized = tilize_to_list(ones_padded)
        ones_tt = ttnn.Tensor(
            ones_tilized,
            [1, 1, 32, bn_size],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )

        running_mean_left = ttnn.mul(ttnn.sub(ones_tt, momentum_tt), running_mean)
        mean_reshaped = mean.view(1, 1, 32, 32)
        mean_tilized = tilize_to_list(mean_reshaped)
        mean_tt = ttnn.Tensor(
            mean_tilized,
            [1, 1, H, W],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )
        running_mean_right = ttnn.mul(momentum_tt, mean_tt)
        running_mean = ttnn.add(running_mean_left, running_mean_right)

        running_var_left = ttnn.mul(ttnn.sub(ones_tt, momentum_tt), running_var)
        running_var_right = ttnn.mul(momentum_tt, var_tt)
        running_var = ttnn.add(running_var_left, running_var_right)

        gamma_tor = gamma.cpu().to_torch()
        gamma_tor = torch.Tensor(gamma_tor).reshape((1, 1, H, W))
        gamma_tor = untilize(gamma_tor)

        beta_tor = beta.cpu().to_torch()
        beta_tor = torch.Tensor(beta_tor).reshape((1, 1, H, W))
        beta_tor = untilize(beta_tor)

        x_gamma_tor = torch.mul(x_div_sqrt_tor, gamma_tor)

        Y_tor = torch.add(x_gamma_tor, beta_tor)

        Y_tilized = tilize_to_list(Y_tor)
        Y = ttnn.Tensor(
            Y_tilized,
            [batch_size, 1, H, W],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )

    return Y, running_mean, running_var


class ttBatchNorm:
    def __init__(
        self,
        bn_size,
        gamma=None,
        beta=None,
        running_mean=None,
        running_var=None,
        epsilon=1e-5,
        momentum=0.1,
        device=None,
    ):
        if (gamma == None) | (beta == None) | (running_mean == None) | (running_var == None):
            # The scale parameter and the shift parameter (model parameters) are initialized to 1 and 0, respectively
            zeros_torch = torch.tensor([[[bn_size * [0.0]]]])
            zeros_padded = pad_activation(zeros_torch)
            zeros_tilized = tilize_to_list(zeros_padded)
            zeros_tt = ttnn.Tensor(
                zeros_tilized,
                [1, 1, 32, bn_size],
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                device,
            )
            ones_torch = torch.tensor([[[bn_size * [1.0]]]])
            ones_padded = pad_activation(ones_torch)
            ones_tilized = tilize_to_list(ones_padded)
            ones_tt = ttnn.Tensor(
                ones_tilized,
                [1, 1, 32, bn_size],
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                device,
            )

            self.gamma = ones_tt
            self.beta = zeros_tt

            # The variables that are not model parameters are initialized to 0 and 1
            self.running_mean = zeros_tt
            self.running_var = ones_tt
            self.mode = "train"

        else:
            self.gamma = gamma
            self.beta = beta
            self.running_mean = running_mean
            self.running_var = running_var
            self.mode = "inference"

        self.epsilon = epsilon
        self.momentum = momentum
        self.bn_size = bn_size
        self.device = device

    def forward(self, X):
        # Save the updated moving_mean and moving_var
        Y, self.running_mean, self.running_var = tt_batch_norm(
            X,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            eps=self.epsilon,
            momentum=self.momentum,
            bn_size=self.bn_size,
            mode=self.mode,
            device=self.device,
        )
        print(
            "Y:",
            Y,
            "\nmoving_mean:",
            self.running_mean,
            "\nmoving_var:",
            self.running_var,
            "\n",
        )
        return Y, self.running_mean, self.running_var


# pytorch
class PytorchBatchNorm1D(nn.Module):
    def __init__(self, input_dim):
        super(PytorchBatchNorm1D, self).__init__()

        self.batchnorm1d_1 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        bn1_out = self.batchnorm1d_1(x)

        return bn1_out


# run
def run_batchnorm_forward(device, bn_size):
    epsilon = 1e-5

    inputs = torch.FloatTensor(2, bn_size).uniform_(-1.0, 1.0).requires_grad_(True)
    # torch
    bn_torch = PytorchBatchNorm1D(bn_size)
    bn_torch.train()
    # weight_bn = torch.nn.Parameter(torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(True))
    # bias_bn =  torch.nn.Parameter(torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(True))
    # running_mean = torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(False)
    # running_var = torch.FloatTensor(bn_size).uniform_(0., 1.).requires_grad_(False)  #must be positive

    weight_bn = torch.nn.Parameter(torch.ones([bn_size]).requires_grad_(True))
    bias_bn = torch.nn.Parameter(torch.zeros([bn_size]).requires_grad_(True))
    running_mean = torch.zeros([bn_size]).requires_grad_(False)
    running_var = torch.ones([bn_size]).requires_grad_(False)  # must be positive

    bn_torch.batchnorm1d_1.weight = weight_bn
    bn_torch.batchnorm1d_1.bias = bias_bn
    bn_torch.batchnorm1d_1.running_mean = running_mean
    bn_torch.batchnorm1d_1.running_var = running_var
    bn_torch.batchnorm1d_1.eps = epsilon

    # tt
    weight_bn_src = weight_bn.view(1, 1, 1, bn_size)
    gamma_padded = pad_weight(weight_bn_src)
    gamma_untilized = tilize_to_list(gamma_padded)
    gamma_tt = ttnn.Tensor(
        gamma_untilized,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    bias_bn_src = bias_bn.view(1, 1, 1, bn_size)
    beta_padded = pad_weight(bias_bn_src)
    beta_tilized = tilize_to_list(beta_padded)
    beta_tt = ttnn.Tensor(
        beta_tilized,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_mean_bn_src = running_mean.view(1, 1, 1, bn_size)
    running_mean_padded = pad_activation(running_mean_bn_src)
    running_mean_tilized = tilize_to_list(running_mean_padded)
    running_mean_tt = ttnn.Tensor(
        running_mean_tilized,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    running_var_bn_src = running_var.view(1, 1, 1, bn_size)
    running_var_padded = pad_activation(running_var_bn_src)
    running_var_tilized = tilize_to_list(running_var_padded)
    running_var_tt = ttnn.Tensor(
        running_var_tilized,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    epsilon_torch = torch.tensor([[[bn_size * [epsilon]]]])
    epsilon_padded = pad_activation(epsilon_torch)
    epsilon_tilized = tilize_to_list(epsilon_padded)
    eps_tt = ttnn.Tensor(
        epsilon_tilized,
        [1, 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    inputs_torch = inputs.view(inputs.shape[0], 1, 1, bn_size)
    inputs_padded = pad_activation(inputs_torch)
    inputs_tilized = tilize_to_list(inputs_padded)
    inputs_tt = ttnn.Tensor(
        inputs_tilized,
        [inputs_padded.shape[0], 1, 32, bn_size],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    # run through models
    output_bn_torch = bn_torch(inputs)
    bn_tt = ttBatchNorm(bn_size, device=device)
    output_bn_tt, _, _ = bn_tt.forward(inputs_tt)

    output_bn_tt_untilized = untilize(torch.Tensor(output_bn_tt.cpu().to_torch()))
    output_bn_tt_untilized = output_bn_tt_untilized[0, 0, 0, :]

    print("pytorch_out:", output_bn_torch[0][0:10])
    print("tt_out:", output_bn_tt_untilized[0:10])

    test_results, output = comp_allclose_and_pcc(output_bn_torch[0], output_bn_tt_untilized)

    print("\n\n", "atol/rtol:", test_results, "| pcc:", output, "\n\n")

    pcc = comp_pcc(output_bn_torch[0], output_bn_tt_untilized)
    assert float(pcc[1]) > 0.99, f"pcc is lower than 0.99: {float(pcc[1])}"


def test_batchnorm_inference(device):
    run_batchnorm_forward(device, 32)
