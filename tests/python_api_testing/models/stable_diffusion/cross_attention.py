import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
from torch import nn
from torch.nn import functional as F


from pymetal import ttmetal as ttm
from utility_functions import tilize_to_list, print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as TtSoftmax


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        print("weight", weight.shape)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output

class TtCrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, device, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.q_proj_weight = torch.ones([1, 1, d_embed, d_embed]).flatten().tolist()
        self.k_proj_weight = torch.ones([1, 1, d_cross, d_embed]).flatten().tolist()
        self.v_proj_weight = torch.ones([1, 1, d_cross, d_embed]).flatten().tolist()
        self.out_proj_weight = torch.ones([1, 1, d_embed, d_embed]).flatten().tolist()

        self.q_proj_bias = torch.zeros([1, 1, d_embed, d_embed]).flatten().tolist()
        self.k_proj_bias = torch.zeros([1, 1, d_cross, d_embed]).flatten().tolist()
        self.v_proj_bias = torch.zeros([1, 1, d_cross, d_embed]).flatten().tolist()
        self.out_proj_bias = torch.zeros([1, 1, d_embed, d_embed]).flatten().tolist()



        self.q_proj = TtLinear(d_embed, d_embed, self.q_proj_weight, self.q_proj_bias, device)
        self.k_proj = TtLinear(d_cross, d_embed, self.k_proj_weight, self.k_proj_bias, device)
        self.v_proj = TtLinear(d_cross, d_embed, self.v_proj_weight, self.v_proj_bias, device)
        self.out_proj = TtLinear(d_embed, d_embed, self.out_proj_weight, self.out_proj_bias, device)



    def forward(self, x, y):
        input_shape = x.shape()


        _, batch_size, sequence_length, d_embed = input_shape
        # batch_size = 2
        # sequence_length = 1024
        # d_embed = 640

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        print("tt x.shape", x.shape())
        print("tt y.shape", y.shape())



        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = ttm.tensor.reshape(q, *interim_shape)
        k = ttm.tensor.reshape(k, *interim_shape)
        v = ttm.tensor.reshape(v, *interim_shape)

        print("tt q.shape", q.shape())
        print("tt k.shape", k.shape())

        q = ttm.tensor.transpose(q)
        k = ttm.tensor.transpose(k)
        v = ttm.tensor.transpose(v)

        kt = ttm.tensor.transpose(k)



        print("kt shape", kt.shape())
        weight = ttm.tensor.bmm(q, kt)

        dsqrt = math.sqrt(self.d_head)
        weight_shape = weight.shape()

        dsqrt_tensor = torch.ones(weight_shape) * 1/dsqrt
        tt_dsqrt = ttm.tensor.Tensor(tilize_to_list(dsqrt_tensor), dsqrt_tensor.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)



        weight = ttm.tensor.mul(weight, tt_dsqrt)
        weight = TtSoftmax(weight)

        output = ttm.tensor.bmm(weight, v)
        output = ttm.tensor.transpose(output)
        output = ttm.tensor.reshape(output, *input_shape)
        output = self.proj_out(output)
        return output



def run_cross_attention_inference(device):


    n_heads = 8
    d_head = 80
    d_embed = 640
    d_embed = 1280
    n_embeds = 768
    d_cross = 768
    x_shape =  [2, 1024, 640]
    x_shape = [2, 256, 1280]
    # D = 77, this should be 96
    D = 96
    y_shape = [2, D, n_embeds]

    #  n_heads, d_embed, d_cross,

    x = torch.randn(x_shape)
    y = torch.randn(y_shape)

    torch_ca = CrossAttention(n_heads, d_embed, d_cross)
    torch_out = torch_ca(x, y)
    tt_x_shape = [1, 2, 1024, 640]
    tt_x_shape = [1, 2, 256, 1280]
    tt_y_shape = [1, 2, D, n_embeds]

    tt_x = torch.randn(tt_x_shape)
    tt_y = torch.randn(tt_y_shape)

    tt_x = ttm.tensor.Tensor(tilize_to_list(tt_x), tt_x_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    tt_y = ttm.tensor.Tensor(tilize_to_list(tt_y), tt_y_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_ca = TtCrossAttention(n_heads, d_embed, d_cross, device)
    tt_out = tt_ca(tt_x, tt_y)
    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(pytorch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)





if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_cross_attention_inference(device)
    ttm.device.CloseDevice(device)
