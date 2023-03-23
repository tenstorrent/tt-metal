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

from pymetal import ttmetal as ttm
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize
from python_api_testing.fused_ops.linear import Linear as TtLinear


class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.linear_1.weight.data.fill_(1)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd, bias=False)
        self.linear_2.weight.data.fill_(1)

    def forward(self, x):
        x = self.linear_1(x)
        # x = F.silu(x)
        x = F.relu(x)
        x = self.linear_2(x)
        return x


class TtTimeEmbedding(torch.nn.Module):
    def __init__(self,  n_embd, device, state_dict=None):
        super().__init__()
        # Note: Load Weights
        #in_feature = n_embd, out_feature = 4 * n_embd
        weight1_shape = [1, 1, n_embd, 4*n_embd]
        self.linear1_weight = torch.ones(weight1_shape).flatten().tolist()
        self.linear_1 = TtLinear(n_embd, 4*n_embd, self.linear1_weight, bias=None, device=device)
        #in_feature = 4 * n_embd, out_feature = 4 * n_embd
        weight2_shape = [1, 1, 4*n_embd, 4*n_embd]
        self.linear2_weight = torch.ones(weight2_shape).flatten().tolist()
        self.linear_2 = TtLinear(4*n_embd, 4*n_embd, self.linear2_weight, bias=None, device=device)



    def forward(self, x):

        x = self.linear_1(x)
        # this should be a SiLU
        x = ttm.tensor.relu(x)
        x = self.linear_2(x)
        return x



if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    n_embd = 32

    torch.manual_seed(123)

    input_shape =  [1, 1, 32, n_embd]
    input = torch.randn(input_shape)


    torch_emb = TimeEmbedding(n_embd)
    torch_out = torch_emb(input)

    print("pytorch result is ready!,", torch_out.shape)


    # time = torch.reshape(time, [32, 32, 32, 1280])
    # time = torch.randn([32, 32, 32, 1280])
    padded_input = pad_activation(input)
    print("padded input")
    print(padded_input.shape)
    tt_input = ttm.tensor.Tensor(tilize_to_list(padded_input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_emb = TtTimeEmbedding(n_embd, device)
    tt_out = tt_emb(tt_input)
    tt_out = tt_out.to(host).data()

    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))
    diff = (abs(torch_out - tt_untilized_output) < 0.1).all().item()

    if not diff:
        print("bad results")


    # compare results!
    # in_channel

    print(tt_untilized_output.shape, torch_out.shape)
    ttm.device.CloseDevice(device)


    # enable_compile_cache()
    # enable_binary_cache()
