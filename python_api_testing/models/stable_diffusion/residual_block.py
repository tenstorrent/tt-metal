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


class ResidualBlock(torch.nn.Module):
    #https://github.com/kjsman/stable-diffusion-pytorch/blob/8c6faa1b87e545b5ab840491f1b7952d803f54ef/stable_diffusion_pytorch/diffusion.py
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(1, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        return feature
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class TtResidualBlock(torch.nn.Module):
    def __init__(self,  in_channels, out_channels, n_time=1280, state_dict=None):
        super().__init__()
        # Note: Only caring about cases where in_channels == out_channels

        # Extract params from state dict
        # if state_dict != None:
        #     fc1_weight = pad_weight(state_dict["fc1.weight"])
        #     fc1_bias = pad_weight(state_dict["fc1.bias"])


        # else:

        #     fc1_weight = pad_weight(state_dict["fc1.weight"])
        #     fc1_bias = pad_weight(state_dict["fc1.bias"])

        # # Get shapes
        # fc1_weight_shape = fc1_weight.shape



        # # Tilize params
        # fc1_weight = tilize_to_list(fc1_weight)
        # fc1_bias = tilize_to_list(fc1_bias)


        ####### what to implement!

        self.torch_groupnorm_feature = nn.GroupNorm(1, in_channels)
        self.torch_conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.linear_time = nn.Linear(n_time, out_channels)

        # self.linear_time = TtLinear(fc1_weight, *fc1_weight_shape[-2:], fc1_bias, device)

        self.torch_groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.torch_conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # if in_channels == out_channels:
        #     self.residual_layer = nn.Identity()
        # else:
        #     self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        ####### What to implement as residual block!







    def SiLU(self, x):
        # return x * sigmoid(x)
        # return x * (1/(1 + exp(-x)))
        return ttm.tensor.relu(x)

    def forward(self, feature, time):
        residue = feature # make a copy
        # move to cpu and
        feature = feature.to(host).data() # move to cpu

        # print(len(feature))
        feature = torch.Tensor(feature)
        # feature = torch.reshape(feature, [2, 320, 64, 64])
        feature = torch.reshape(feature, [1, 1, 32, 32]) # this should be untilize
        feature = self.torch_groupnorm_feature(feature)
        return feature

        # exec group norm on cpu
        # move from cpu to tensix
        feature = feature.to(device).data() # move to tensix
        feature = self.SiLU(feature)
        # move to cpu again
        # exec conv_feature
        feature = feature.to(host).data()  # move to cpu
        feature = self.torch_conv_feature(feature)
        # move from CPU to tensix
        feature = feature.to(device).data()


        # all on tensix
        time = self.SiLU(time)
        # time = self.linear_time(time)

        time = time.to(host).data()
        time.unsqueeze(-1).unsqueeze(-1)
        time = time.to(device) # to tensix
        merged == ttm.tensor.add(feature, time)
        # merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        # move from tensix to CPU
        merged = merged.to(host).data()
        merged = self.groupnorm_merged(merged)
        merged = merged.to(device).data()
        # move back to tensix
        merged = self.SiLU(merged)
        # move from tensix to CPU
        merged = merged.to(host).data() # move to CPU
        merged = self.conv_merged(merged)
        merged = merged.to(device).data()

        return ttm.tensor.add(merged, residue)

        # return merged + self.residual_layer(residue)




if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    in_channel = 320
    out_channel = 320
    time_shape = [1, 1280]
    feature_shape = [2, 320, 64, 64]

    feature_shape = [1, 32, 32, 32]

    torch.manual_seed(123)
    time = torch.randn(time_shape)
    feature = torch.randn(feature_shape)


    torch_rb = ResidualBlock(in_channel, out_channel)
    torch_out = torch_rb(feature, time)

    print("pytorch result is ready!")


    # time = torch.reshape(time, [32, 32, 32, 1280])
    # time = torch.randn([32, 32, 32, 1280])
    time = torch.randn([1, 1, 32, 32])
    new_time_shape = [1, 1, 32, 32]
    tt_feature = ttm.tensor.Tensor(tilize_to_list(feature), feature_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_time = ttm.tensor.Tensor(tilize_to_list(time), new_time_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_rb = TtResidualBlock(in_channel, out_channel)
    tt_out = tt_rb(tt_feature, tt_time)

    diff = (abs(torch_out - tt_out) < 0.1).all().item()

    if not diff:
        print("bad results")


    # compare results!
    # in_channel


    ttm.device.CloseDevice(device)


    # enable_compile_cache()
    # enable_binary_cache()
