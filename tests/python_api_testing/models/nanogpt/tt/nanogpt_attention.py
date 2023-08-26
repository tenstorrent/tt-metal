import torch
from torch.nn import functional as F
import torch.nn as nn
import tt_lib
from dataclasses import dataclass
import math
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.nanogpt.tt.nanogpt_config import GPTConfig
from python_api_testing.models.helper_funcs import Linear


from transformers import GPT2LMHeadModel

from tt_models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

class TtCausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig(), state_dict, base_address, device):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config

        # key, query, value projections for all heads, but in a batch
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)


        self.device = device
        # Get the weights
        self.tt_weight_c_attn = state_dict[f"{base_address}.c_attn.weight"]
        self.tt_weight_c_proj = state_dict[f"{base_address}.c_proj.weight"]

        # Push weights to Ttp device
        self.tt_weight_c_attn = torch2tt_tensor(
            self.tt_weight_c_attn, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.tt_weight_c_proj = torch2tt_tensor(
            self.tt_weight_c_proj, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.tt_weight_c_attn = tt_lib.tensor.transpose(self.tt_weight_c_attn)
        self.tt_weight_c_proj = tt_lib.tensor.transpose(self.tt_weight_c_proj)

        # Load biases
        self.tt_bias_c_attn = torch2tt_tensor(
            state_dict[f"{base_address}.c_attn.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.tt_bias_c_proj = torch2tt_tensor(
            state_dict[f"{base_address}.c_proj.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.dropout = self.config.dropout

        self.register_buffer("bias", torch.tril(torch.ones(self.config.block_size, self.config.block_size)).view(1, 1, self.config.block_size, self.config.block_size))

        self.c_attn = Linear(self.config.n_embd, 3 * config.n_embd, self.tt_weight_c_attn, self.tt_bias_c_attn)
        self.c_proj = Linear(self.config.n_embd, self.config.n_embd, self.tt_weight_c_proj, self.tt_bias_c_proj)


    def forward(self, x):

        _, B, T, C = x.shape() # batch size, sequence length, embedding dimensionality (n_embd)

        x1 = self.c_attn(x)

        pt_x1 = tt2torch_tensor(x1)
        pt_x1 = pt_x1.squeeze(0)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = pt_x1.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        tt_att = torch_to_tt_tensor_rm(att, self.device, put_on_device=False)

        tt_att = fallback_ops.softmax(tt_att, dim=-1)

        att = tt2torch_tensor(tt_att)
        att = self.attn_dropout(att)

        tt_att = torch_to_tt_tensor_rm(att, self.device, put_on_device=False)

        tt_v = torch_to_tt_tensor_rm(v, self.device, put_on_device=False)

        #y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        tt_y = tt_lib.tensor.bmm(tt_att, tt_v)

        #y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        tt_y = tt_lib.tensor.transpose_hc(tt_y)
        tt_y = fallback_ops.reshape(tt_y, 1, B, T, C)


        # output projection
        x2 = self.c_proj(tt_y)

        pt_x2 = tt2torch_tensor(x2)

        y = self.resid_dropout(pt_x2)
        y = torch2tt_tensor(y, self.device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
        return y
