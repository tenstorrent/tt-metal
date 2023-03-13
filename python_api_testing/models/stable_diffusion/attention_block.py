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
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as TtLinear


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)

        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.linear_geglu_2.weight.data.fill_(1)
        self.linear_geglu_2.bias.data.fill_(0)

        self.linear_geglu_2.weight.data.fill_(1)
        self.linear_geglu_1.bias.data.fill_(0)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)
        return x
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))   # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)  # (n, c, hw)
        x = x.view((n, c, h, w))    # (n, c, h, w)

        return self.conv_output(x) + residue_long


class TtAttentionBlock(torch.nn.Module):
    def __init__(self,  n_head: int, n_embd: int, d_context=768, state_dict=None):
        super().__init__()

        channels = n_head * n_embd

        self.torch_groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.torch_conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.torch_layernorm_1 = nn.LayerNorm(channels)
        self.torch_attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.torch_layernorm_2 = nn.LayerNorm(channels)
        self.torch_attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.torch_layernorm_3 = nn.LayerNorm(channels)
        # in_features, out_features, weight, bias, device
        # TODO: fill in weights and bias
        self.linear_geglu_1 = TtLinear(channels, 4 * channels * 2)

        self.linear_geglu_2 = TtLinear(4 * channels, channels)


        self.torch_conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)


    def forward(self, x, context):
        residue_long = x

        input_shape = [2, 320, 64, 64]
        x = torch.Tensor(x.to(host).data()).reshape(input_shape)

        x = untilize(x)
        return self.torch_groupnorm(x)





if __name__ == "__main__":
    # 20 channels
    # 8 heads
    # 40 embds
    # input: 2, 320, 64, 64


    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    n_head = 8
    n_embd = 40
    channels = 20
    input_shape = [2, 320, 64, 64]


    torch.manual_seed(123)
    input = torch.randn(input_shape)

    # TODO: context!
    context = None
    tt_context = None

    torch_attention = AttentionBlock(n_head, n_embd)
    torch_out = torch_attention(input)

    print("pytorch result is ready!")


    # time = torch.reshape(time, [32, 32, 32, 1280])
    # time = torch.randn([32, 32, 32, 1280])

    tt_input = ttm.tensor.Tensor(tilize_to_list(input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    # TODO: tt_context


    tt_attention = TtAttentionBlock(n_head, n_embd)
    tt_out = tt_attention(tt_input, tt_context)

    diff = (abs(torch_out - tt_out) < 0.1).all().item()
    print_diff_argmax(tt_got_back, ref_lnorm)
    if not diff:
        print("bad results")


    # compare results!
    # in_channel


    ttm.device.CloseDevice(device)


    # enable_compile_cache()
    # enable_binary_cache()
