# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
import math
from models.helper_funcs import Linear

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtCausalSelfAttention(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.block_size = 1024

        self.device = device
        # Get the weights
        self.tt_weight_c_attn = ttnn.load_tensor(tt_cache_path + base_address + ".c_attn.weight" + str(dtype) + ".bin")

        self.tt_weight_c_proj = ttnn.load_tensor(tt_cache_path + base_address + ".c_proj.weight" + str(dtype) + ".bin")

        self.tt_weight_c_attn = ttnn.transpose(self.tt_weight_c_attn, -2, -1)
        self.tt_weight_c_proj = ttnn.transpose(self.tt_weight_c_proj, -2, -1)

        # Load biases
        self.tt_bias_c_attn = ttnn.load_tensor(tt_cache_path + base_address + ".c_attn.bias" + str(dtype) + ".bin")

        self.tt_bias_c_proj = ttnn.load_tensor(tt_cache_path + base_address + ".c_proj.bias" + str(dtype) + ".bin")

        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd

        temp_bias = ttnn.tril(ttnn.ones([1, 1, self.block_size, self.block_size]))
        temp_bias = tt_to_torch_tensor(temp_bias)
        self.register_buffer(
            "bias",
            temp_bias,
        )

        self.c_attn = Linear(
            self.config.n_embd,
            3 * config.n_embd,
            self.tt_weight_c_attn,
            self.tt_bias_c_attn,
        )
        self.c_proj = Linear(
            self.config.n_embd,
            self.config.n_embd,
            self.tt_weight_c_proj,
            self.tt_bias_c_proj,
        )

    def const_tensor(self, shape, value):
        return ttnn.full(shape, value)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        (
            _,
            B,
            T,
            C,
        ) = x.get_legacy_shape()  # batch size, sequence length, embedding dimensionality (n_embd)

        x1 = self.c_attn(x)

        pt_x1 = tt_to_torch_tensor(x1)
        pt_x1 = pt_x1.squeeze(0)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = pt_x1.split(self.n_embd, dim=2)

        k = torch_to_tt_tensor_rm(k, self.device)
        k = ttnn.reshape_on_device(k, B, T, self.n_head, C // self.n_head)
        k = ttnn.transpose(k, 1, 2)

        q = torch_to_tt_tensor_rm(q, self.device)
        q = ttnn.reshape_on_device(q, B, T, self.n_head, C // self.n_head)
        q = ttnn.transpose(q, 1, 2)

        v = torch_to_tt_tensor_rm(v, self.device)

        v = ttnn.reshape_on_device(v, B, T, self.n_head, C // self.n_head)
        v = ttnn.transpose(v, 1, 2)

        # manual implementation of attention
        key_layer_transposed = ttnn.transpose(k, -2, -1)
        att = ttnn.matmul(q, key_layer_transposed)

        const_att = self.const_tensor(att.get_legacy_shape(), 1.0 / math.sqrt(k.get_legacy_shape()[-1]))

        att = ttnn.mul(att, const_att)

        att = tt_to_torch_tensor(att)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        tt_att = torch_to_tt_tensor_rm(att, self.device, put_on_device=False)

        tt_att = ttnn.softmax(tt_att)  # Using ttnn.softmax reduces pcc from 0.99 to 0.98 for whole model

        tt_y = ttnn.matmul(tt_att, v)

        tt_y = ttnn.transpose(tt_y, 1, -2)
        tt_y = ttnn.reshape_on_device(tt_y, 1, B, T, C)

        # output projection
        x2 = self.c_proj(tt_y)
        return x2
