# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib
from tt_lib.fallback_ops import fallback_ops
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        parameters,
        device,
        dim,
        window_size,
        shift_size,
        num_heads,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.parameters = parameters
        self.device = device
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.relative_position_bias_table = torch.zeros(
            (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        relative_position_bias = ttnn.from_torch(relative_position_bias, ttnn.bfloat16)
        relative_position_bias = ttnn.to_device(relative_position_bias, self.device)
        relative_position_bias = ttnn.reshape(relative_position_bias, (N, N, -1))
        relative_position_bias = ttnn.to_layout(relative_position_bias, ttnn.ROW_MAJOR_LAYOUT)
        relative_position_bias = ttnn.from_device(relative_position_bias)
        relative_position_bias = ttnn.to_torch(relative_position_bias)
        relative_position_bias = torch.permute(relative_position_bias, (2, 0, 1))
        relative_position_bias = relative_position_bias.unsqueeze(0)
        relative_position_bias = ttnn.from_torch(
            relative_position_bias, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        logit_scale = None

        B, H, W, C = x.get_legacy_shape()
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_values = (0, 0, 0, pad_r, 0, pad_b)
        x = fallback_ops.pad(x, pad_values)
        _, pad_H, pad_W, _ = x.get_legacy_shape()

        self.shift_size = self.shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        x = tt_to_torch_tensor(x)
        # cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        x = torch_to_tt_tensor_rm(x, self.device)

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        x = tt_to_torch_tensor(x)
        x = x.view(
            B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, self.window_size[0] * self.window_size[1], C)
        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)

        qkv_weight = self.parameters.qkv.weight
        qkv_bias = self.parameters.qkv.bias

        # multi-head attention
        if logit_scale is not None and qkv_bias is not None:
            qkv_bias = tt_lib.tensor.clone(qkv_bias)
            shape = qkv_bias.get_legacy_shape()
            numel = shape[0] * shape[1] * shape[2] * shape[3]
            length = numel // 3
            qkv_bias[length : 2 * length] = 0

        # tt to torch and torch to ttnn of qkv_weight
        qkv_weight = tt_to_torch_tensor(qkv_weight)
        qkv_weight = ttnn.from_torch(qkv_weight, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # tt to torch and torch to ttnn of qkv_bias
        qkv_bias = tt_to_torch_tensor(qkv_bias)
        qkv_bias = ttnn.from_torch(qkv_bias, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # tt to torch and torch to ttnn of x
        x = tt_to_torch_tensor(x)
        x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        qkv_weight = ttnn.to_layout(qkv_weight, ttnn.TILE_LAYOUT)
        qkv_bias = ttnn.to_layout(qkv_bias, ttnn.TILE_LAYOUT)

        qkv = ttnn.linear(x, qkv_weight, bias=qkv_bias)
        # ttnn to torch and torch to tt of qkv_weight and x
        qkv = ttnn.to_torch(qkv)
        qkv = qkv.squeeze(0)
        x = ttnn.to_torch(x)
        x = x.squeeze(0)

        qkv = qkv.reshape(x.size(0), x.size(1), 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if logit_scale is not None:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
        else:
            q = q * (C // self.num_heads) ** -0.5
            attn = q.matmul(k.transpose(-2, -1))
        # add relative position bias
        attn = ttnn.from_torch(attn, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        attn = ttnn.add(attn, relative_position_bias)

        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)
        if sum(self.shift_size) > 0:
            # generate attention mask
            attn_mask = tt_lib.tensor.zeros([pad_H, pad_W], data_type=x.data_type, device=x.device)
            h_slices = (
                (0, -self.window_size[0]),
                (-self.window_size[0], -self.shift_size[0]),
                (-self.shift_size[0], None),
            )
            w_slices = (
                (0, -self.window_size[1]),
                (-self.window_size[1], -self.shift_size[1]),
                (-self.shift_size[1], None),
            )
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = fallback_ops.reshape(
                attn_mask,
                (pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1]),
            )
            attn_mask = tt_lib.tensor.permute(attn_mask, (0, 2, 1, 3))
            attn_mask = fallback_ops.reshape(attn_mask, (num_windows, self.window_size[0] * self.window_size[1]))
            attn_mask = tt_to_torch_tensor(attn_mask)
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = ttnn.to_torch(attn)
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, x.size(1), x.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x.size(1), x.size(1))
            attn = ttnn.from_torch(attn, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        attn = ttnn.softmax(attn, dim=-1)
        attn = ttnn.to_torch(attn)
        attn = F.dropout(attn, p=self.attention_dropout, training=self.training)
        attn = ttnn.from_torch(attn, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        v = ttnn.from_torch(v, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        x = ttnn.matmul(attn, v)
        x = ttnn.to_torch(x)
        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)
        x = tt_lib.tensor.transpose(x, 2, 1)

        x = tt_to_torch_tensor(x)
        x = x.reshape(x.size(0), x.size(1), C)
        x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # tt to torch and torch to ttnn of proj_weight
        proj_weight = tt_to_torch_tensor(self.parameters.proj.weight)
        proj_weight = ttnn.from_torch(
            proj_weight, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        # tt to torch and torch to ttnn of proj_bias
        proj_bias = tt_to_torch_tensor(self.parameters.proj.bias)
        proj_bias = ttnn.from_torch(proj_bias, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        proj_weight = ttnn.to_layout(proj_weight, ttnn.TILE_LAYOUT)
        proj_bias = ttnn.to_layout(proj_bias, ttnn.TILE_LAYOUT)

        x = ttnn.linear(x, proj_weight, bias=proj_bias)
        x = ttnn.to_torch(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # reverse windows
        x = x.view(
            B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :]
        x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return x
