import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pymetal import ttlib as ttl
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float
from utils import move_to_device, move_to_cpu
from python_api_testing.fused_ops.linear import Linear as TtLinear
from pymetal.ttlib.fused_ops.layernorm import Layernorm as TtLayerNorm

def torch_group_norm(num_groups, num_channels):
    torch_gn = torch.nn.GroupNorm(num_groups, num_channels, affine=True)
    print(torch_gn.weight.shape, "weight shape")
    print(torch_gn.bias.shape, "bias shape")
    return torch_gn




class TtGroupNorm2D(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, epsf:float = 1e-4, device=None, host=None, state_dict=None, base_address=None):
        super().__init__()

        self.num_groups = num_groups
        assert num_channels % num_groups == 0, f"num_channels: {num_channels}, num_groups: {num_groups}"
        self.chunks = num_channels // num_groups

        weights_ = pad_weight(torch.ones((1, num_channels, 1, 1)))
        bias_ = pad_weight(torch.zeros((1, num_channels, 1, 1)))

        chunked_weight = torch.chunk(weights_, num_groups)
        chunked_bias = torch.chunk(bias_, num_groups)

        self.weight = [move_to_device(c, device) for c in chunked_weight]
        self.bias = [move_to_device(c, device) for c in chunked_bias]
        self.device = device
        self.host = host
        self.epsf = epsf


    def forward(self, input):
        input_shape = input.shape()
        input = move_to_cpu(input, self.host)


        input_chunked = torch.chunk(input, self.chunks, dim=1)

        normalized = []
        for index, chunk in enumerate(input_chunked):
            tt_chunk = move_to_device(chunk, self.device)
            layernorm_f = TtLayerNorm(self.weight[index], self.bias[index], self.epsf, input_shape[-2], input_shape[-1], self.device, num_dims=2)
            tt_chunk - layernorm_f(tt_chunk)
            normalized.append(move_to_cpu(tt_chunk, self.host))

        concated_torch = torch.concat(normalized, dim=1)

        return move_to_device(concated_torch, self.device)




def run_group_norm2d_inference(device, host):
    num_groups = 32
    num_channels = 320
    epsf = 1e-4
    input_shape =  [2, num_channels, 64, 64]
    input = torch.randn(input_shape)

    torch_gn = torch_group_norm(num_groups, num_channels)
    torch_out = torch_gn(input)

    tt_input = ttl.tensor.Tensor(tilize_to_list(input), input_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    tt_gn = TtGroupNorm2D(num_groups, num_channels, device=device, host=host)
    tt_out = tt_gn(tt_input).to(host).data()
    tt_out = torch.Tensor(tt_out).reshape(torch_out.shape)
    tt_untilized_output = untilize(tt_out)
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_group_norm2d_inference(device, host)
    ttl.device.CloseDevice(device)
