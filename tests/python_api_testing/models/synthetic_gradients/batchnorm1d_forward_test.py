import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"

sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torch import nn
from torchvision import transforms, datasets
import libs

from libs import tt_lib as ttl
from models.utility_functions import tilize_to_list, untilize, comp_allclose_and_pcc, tt2torch, torch2tt_tensor
from libs.tt_lib.utils import pad_activation, pad_weight

epsilon = 1e-5


def batch_norm(X, gamma, beta, running_mean, running_var, eps, momentum, inference = False):
    # Use is_grad_enabled to determine whether we are in training mode
    if inference == True:
        print('inference mode')
        # In prediction mode, use mean and variance obtained by moving average
        var_plus_eps = ttl.tensor.add(epsilon, running_var)
        sqrt_var = ttl.tensor.sqrt(var_plus_eps)
        sqrt_inv = ttl.tensor.recip(sqrt_var)
        x_minus_mean = ttl.tensor.sub(X, running_mean)
        X_hat = ttl.tensor.mul(x_minus_mean, sqrt_inv)

    else:
        print('train mode')
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * running_mean + momentum * mean
        moving_var = (1.0 - momentum) * running_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    print(X_hat, gamma, beta, Y)
    return Y, moving_mean.data, moving_var.data






class ttBatchNorm1d_FW():
    def __init__(self, gamma, beta, moving_mean, moving_var, epsilon):
        super().__init__()
        self.gamma = nn.Parameter(gamma)
        self.beta = nn.Parameter(beta)
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        self.epsilon = epsilon
    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=self.epsilon, momentum=0.1)
        print('Y:', Y,'moving_mean:', self.moving_mean, 'moving_var:', self.moving_var, '\n')
        return Y









def batchnorm1d_forward(X, gamma, beta, moving_mean, moving_var, eps, momentum):
        mean = X.mean(dim=0)
        var = ((X - mean) ** 2).mean(dim=0)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
        Y = gamma * X_hat + beta  # Scale and shift
        print(X_hat, gamma, beta, Y)
        return Y, moving_mean.data, moving_var.data

class PytorchBatchNorm1D(nn.Module):
    def __init__(self, input_dim):
        super(PytorchBatchNorm1D, self).__init__()

        self.batchnorm1d_1 = nn.BatchNorm1d(input_dim)

    def forward(self, x):

        bn1_out =  self.batchnorm1d_1(x)

        return bn1_out


def run_btchnorm_forward(bn_size, device):
    host = ttl.device.GetHost()

    inputs = torch.FloatTensor(1, bn_size).uniform_(-1., 1.).requires_grad_(True)
    # torch
    bn_torch = PytorchBatchNorm1D(bn_size)
    # bn_torch.eval()
    weight_bn = torch.nn.Parameter(torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(True))
    bias_bn =  torch.nn.Parameter(torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(True))
    running_mean = torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(False)
    running_var = torch.FloatTensor(bn_size).uniform_(0., 1.).requires_grad_(False)  #must be positive

    bn_torch.batchnorm1d_1.weight = weight_bn
    bn_torch.batchnorm1d_1.bias = bias_bn
    bn_torch.batchnorm1d_1.running_mean = running_mean
    bn_torch.batchnorm1d_1.running_var = running_var
    bn_torch.batchnorm1d_1.eps = epsilon

    # tt
    weight_bn_src = weight_bn.view(1, 1, 1, bn_size)

    # weight_bn_tt_new = torch2tt_tensor(weight_bn_src, device)
    gamma_new = pad_weight(weight_bn_src)
    gamma_new_untilized = untilize(torch.Tensor(gamma_new.to(host).data()).reshape(gamma_new.shape()))

    weight_bn_tt = torch.zeros(1, 1, 32, bn_size)
    weight_bn_tt[:, :, :1, :] = weight_bn_src

    tilized_weight_bn_tt = tilize_to_list(weight_bn_tt)
    gamma = ttl.tensor.Tensor(tilized_weight_bn_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    gamma_untilized = untilize(torch.Tensor(gamma.to(host).data()).reshape(gamma.shape()))


    print('gamma new:', gamma_new_untilized)
    print('gamma:', gamma_untilized)


    # bias_bn_src = bias_bn.view(1, 1, 1, bn_size)
    # bias_bn_tt = torch.zeros(1, 1, 32, bn_size)
    # bias_bn_tt[:, :, :1, :] = bias_bn_src
    # tilized_bias_bn_tt= tilize_to_list(bias_bn_tt)
    # beta = ttl.tensor.Tensor(tilized_bias_bn_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    # running_mean_bn_src = running_mean.view(1, 1, 1, bn_size)
    # running_mean_bn_tt = torch.zeros(1, 1, 32, bn_size)
    # running_mean_bn_tt[:, :, :1, :] = running_mean_bn_src
    # tilized_running_mean_tt= tilize_to_list(running_mean_bn_tt)
    # running_mean_tt = ttl.tensor.Tensor(tilized_running_mean_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16,ttl.tensor.Layout.TILE, device)

    # running_var_bn_src = running_var.view(1, 1, 1, bn_size)
    # running_var_bn_tt = torch.zeros(1, 1, 32, bn_size)
    # running_var_bn_tt[:, :, :1, :] = running_var_bn_src
    # tilized_running_var_tt= tilize_to_list(running_var_bn_tt)
    # running_var_tt = ttl.tensor.Tensor(tilized_running_var_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16,ttl.tensor.Layout.TILE, device)

    # epsilon_torch = torch.tensor([[[bn_size*[epsilon]]]])
    # epsilon_tor = torch.zeros(1, 1, 32, bn_size)
    # epsilon_tor[:, :, :1, :] = epsilon_torch
    # tilized_eps_tt= tilize_to_list(epsilon_tor)
    # eps_tt = ttl.tensor.Tensor(tilized_eps_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16,ttl.tensor.Layout.TILE, device)

    # inputs_bn_src = inputs.view(1, 1, 1, bn_size)
    # inputs_bn_tt = torch.zeros(1, 1, 32, bn_size)
    # inputs_bn_tt[:, :, :1, :] = inputs_bn_src
    # tilized_inputs_tt = tilize_to_list(inputs_bn_tt)
    # X_tt = ttl.tensor.Tensor(tilized_inputs_tt, [1, 1,  32, bn_size], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    # # run through models
    # output_bn_torch = bn_torch(inputs)
    # bn_tt =  batchnorm1d_inference(gamma, beta, running_mean_tt, running_var_tt, eps_tt)
    # output_bn_tt = bn_tt(X_tt)

    # output_bn_tt_untilized = untilize(torch.Tensor(output_bn_tt.to(host).data()).reshape(output_bn_tt.shape()))
    # output_bn_tt_untilized = output_bn_tt_untilized[0, 0, 0, :]

    # print('pytorch_out:', output_bn_torch[0][0:10])
    # print('tt_out:', output_bn_tt_untilized[0:10])

    # test_results, output = comp_allclose_and_pcc(output_bn_torch[0], output_bn_tt_untilized)

    # print('\n\n', 'atol/rtol:', test_results, '| pcc:', output, '\n\n')


# def test_batchnorm_inference():
if __name__ == "__main__":

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    run_btchnorm_inference(32, device)
    ttl.device.CloseDevice(device)
