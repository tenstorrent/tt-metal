from pymetal import ttmetal as ttm
from pathlib import Path
import sys
import torch
from torch import nn
f = f"{Path}(__file__).parent"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append('./python_api_testing/models/')
sys.path.append('./python_api_testing/sweep_tests')

from utility_functions import tilize_to_list, untilize
from comparison_funcs import comp_pcc



def ttLinear(weight, bias):

    def linear_(activation):
        weight_T = ttm.tensor.transpose(weight)
        output = ttm.tensor.matmul(activation, weight_T)
        output_plus_bias = ttm.tensor.bcast(output, bias, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.H)
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
    inputs_tt = ttm.tensor.Tensor(tilized_inputs, inputs_targ.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    weight_tt = tilize_to_list(weight_tt)
    bias_tt = tilize_to_list(bias_tt)
    weight_tt = ttm.tensor.Tensor(weight_tt, [1, 1, out_features, in_features], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE,  device )
    bias_tt = ttm.tensor.Tensor(bias_tt, [1, 1, 32, out_features],  ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE,  device)

    linear_tt = ttLinear(in_features, out_features, weight_tt, bias_tt, device)
    output_tt = linear_tt(inputs_tt)
    output_tt = untilize(torch.Tensor(output_tt.to(host).data()).reshape(output_tt.shape()))
    output_tt = output_tt[0, 0, 0, :]

    test_result = comp_pcc(output_torch, output_tt)

    print('\n\n', test_result, '\n\n')


if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_linear_test(1024, 256)
    ttm.device.CloseDevice(device)
