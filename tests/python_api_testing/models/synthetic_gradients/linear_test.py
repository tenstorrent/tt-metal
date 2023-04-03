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

from models.utility_functions import tilize_to_list, untilize
from sweep_tests.comparison_funcs import comp_pcc

def ttLinear(weight, bias):

    def linear_(activation):
        weight_T = ttl.tensor.transpose(weight)
        output = ttl.tensor.matmul(activation, weight_T)
        output_plus_bias = ttl.tensor.bcast(output, bias, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)
        return output_plus_bias

    return linear_

def torchLinear(in_features, out_features, weight, bias):
    linear_torch = torch.nn.Linear(out_features, in_features)
    linear_torch.weight = nn.Parameter(weight)
    linear_torch.bias = nn.Parameter(bias)

    return linear_torch


def run_linear_test(in_features, out_features):
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
    inputs_tt = ttl.tensor.Tensor(tilized_inputs, inputs_targ.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    weight_tt = tilize_to_list(weight_tt)
    bias_tt = tilize_to_list(bias_tt)
    weight_tt = ttl.tensor.Tensor(weight_tt, [1, 1, out_features, in_features], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE,  device )
    bias_tt = ttl.tensor.Tensor(bias_tt, [1, 1, 32, out_features],  ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE,  device)

    linear_tt = ttLinear(weight_tt, bias_tt)
    output_tt = linear_tt(inputs_tt)
    output_tt = untilize(torch.Tensor(output_tt.to(host).data()).reshape(output_tt.shape()))
    output_tt = output_tt[0, 0, 0, :]

    test_result = comp_pcc(output_torch, output_tt)

    print('\n\n', test_result, '\n\n')


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_linear_test(1024, 256)
    ttl.device.CloseDevice(device)
