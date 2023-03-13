import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from pymetal import ttmetal as ttm
from utility_functions import tilize_to_list, print_diff_argmax, untilize, tilize, tilize_to_list
from python_api_testing.fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as TtSoftmax

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.in_proj.weight.data.fill_(1)

        if in_proj_bias:
            self.in_proj.bias.data.fill_(0)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.out_proj.weight.data.fill_(1)
        if out_proj_bias:
            self.out_proj.bias.data.fill_(0)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        _, batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        tt = self.in_proj(x)

        q, k, v = tt.chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output


class TtSelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, device, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj_weight = torch.ones([1, 1, d_embed, 3 * d_embed]).flatten().tolist()
        self.out_proj_weight = torch.ones([1, 1, d_embed, d_embed]).flatten().tolist()
        if in_proj_bias:
            in_proj_bias = torch.zeros([1, 1, d_embed, 3 * d_embed]).flatten().tolist()
        else:
            in_proj_bias = None
        if out_proj_bias:
            out_proj_bias = torch.zeros([1, 1, d_embed, d_embed]).flatten().tolist()
        else:
            out_proj_bias = None
        self.in_proj = TtLinear(d_embed, 3*d_embed, self.in_proj_weight, bias=in_proj_bias, device=device)
        self.out_proj = TtLinear(d_embed, d_embed, self.out_proj_weight, bias=out_proj_bias, device=device)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        # heads 12
        # n_embeds 768
        # x = 1, 77, 768
    def forward(self, x, device, causal_mask=False):
        input_shape = x.shape()
        _, batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        print(interim_shape)
        x = self.in_proj(x)

        _shape = x.shape()
        t_x = x.to(host).data()
        t_x = torch.Tensor(t_x).reshape(_shape)
        t_x = untilize(t_x)
        q, k, v = t_x.chunk(3, dim=-1)


        q = ttm.tensor.Tensor(tilize_to_list(q), q.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
        k = ttm.tensor.Tensor(tilize_to_list(k), k.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
        v = ttm.tensor.Tensor(tilize_to_list(v), v.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)


        # TODO: CHUNK THE INPUT
        # TODO: can we do it on device?
        # q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = ttm.tensor.transpose(q)
        k = ttm.tensor.transpose(k)
        v = ttm.tensor.transpose(v)

        kt = ttm.tensor.transpose(k)
        weight = ttm.tensor.bmm(q, kt)

        # ignore casaul_mask
        dsqrt = math.sqrt(self.d_head)
        weight_shape = weight.shape()
        dsqrt_tensor = torch.ones(weight_shape) * 1/dsqrt
        tt_dsqrt = ttm.tensor.Tensor(tilize_to_list(dsqrt_tensor), dsqrt_tensor.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

        weight = ttm.tensor.mul(weight, tt_dsqrt)

        weight = TtSoftmax(weight)

        output = ttm.tensor.bmm(weight, v)
        output = ttm.tensor.transpose(output)
        output = ttm.tensor.reshape(output, *input_shape)

        output = self.out_proj(output)
        return output





def run_self_attention_inference(device):
    n_heads = 12
    d_embeds = 768
    # D is 77 originally
    D = 96
    input_shape =  [1, 1, 96, d_embeds]
    input = torch.randn(input_shape)
    torch_sa = SelfAttention(n_heads, d_embeds)
    torch_out = torch_sa(input)

    tt_input = ttm.tensor.Tensor(tilize_to_list(input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_sa = TtSelfAttention(n_heads, d_embeds, device)
    tt_out = tt_sa(tt_input, device).to(host).data()
    tt_out = torch.Tensor(tt_out).reshape(torch_out.shape)
    tt_untilized_output = untilize(tt_out)
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)






if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_self_attention_inference(device)
    ttm.device.CloseDevice(device)
