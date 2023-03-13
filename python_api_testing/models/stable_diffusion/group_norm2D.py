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
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float
from python_api_testing.fused_ops.linear import Linear as TtLinear


def torch_group_norm(num_groups, num_channels, input):
    torch_gn = torch.nn.GroupNorm(num_groups, num_channels, affine=False)



def TT_stuff():

    pass




if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    num_groups = 32
    num_channels = 320
    input_shape = [2, num_channels, 64, 64]



    torch.manual_seed(123)
    input = torch.randn(input_shape)


    torch_out = torch_group_norm(num_groups, num_channels, input)

    print("pytorch result is ready!")


    tt_input = ttm.tensor.Tensor(tilize_to_list(input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    # tt_rb = TtResidualBlock(in_channel, out_channel)
    # tt_out = tt_rb(tt_feature, tt_time)

    # diff = (abs(torch_out - tt_out) < 0.1).all().item()

    # if not diff:
    #     print("bad results")


    # compare results!
    # in_channel


    ttm.device.CloseDevice(device)
