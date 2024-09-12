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
from typing import Optional


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
        attn_mask=None,
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
        self.attn_mask = attn_mask

    def forward(self, x):
        relative_position_bias = self.parameters[
            "relative_position_bias"
        ]  # relative position bias is taken from torch since it won't differ from input

        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.from_device(x)
        B, H, W, C = x.get_legacy_shape()
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_values = (0, 0, 0, pad_r, 0, pad_b)
        x = fallback_ops.pad(x, pad_values)  # check this 6 side padding
        _, pad_H, pad_W, _ = x.get_legacy_shape()

        self.shift_size = self.shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        x = tt_to_torch_tensor(x)
        # cyclic shift
        # Torch is used since ttnn doesn't support roll, should check this
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        x = torch_to_tt_tensor_rm(x, self.device)

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])

        # Torch is used since ttnn doesn't support reshape or permute of 6D tensor
        x = tt_to_torch_tensor(x)
        x = x.view(
            (B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C)
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)

        x = ttnn.reshape(x, (B * num_windows, self.window_size[0] * self.window_size[1], C))

        qkv_weight = self.parameters.qkv.weight
        qkv_bias = self.parameters.qkv.bias

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        qkv = ttnn.linear(
            x,
            qkv_weight,
            bias=qkv_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )

        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT)
        qkv = ttnn.from_device(qkv)
        qkv = ttnn.reshape(qkv, (x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads))
        qkv = ttnn.to_device(qkv, device=self.device)
        qkv = ttnn.to_torch(qkv)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = ttnn.from_torch(q, device=self.device, layout=ttnn.TILE_LAYOUT)
        k = ttnn.from_torch(k, device=self.device, layout=ttnn.TILE_LAYOUT)
        v = ttnn.from_torch(v, device=self.device, layout=ttnn.TILE_LAYOUT)

        q = q * (C // self.num_heads) ** -0.5
        k = ttnn.permute(k, (0, 1, 3, 2))
        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )
        # add relative position bias
        attn = ttnn.add(attn, relative_position_bias)

        if sum(self.shift_size) > 0:  # some torch ops have been used should check this function
            attn = attn + self.attn_mask
            # attn = ttnn.from_torch(attn, dtype=ttnn.bfloat16)
            attn = ttnn.from_device(attn)
            attn = ttnn.to_layout(attn, layout=ttnn.ROW_MAJOR_LAYOUT)
            attn = ttnn.reshape(attn, (-1, self.num_heads, x.shape[1], x.shape[1]))
            attn = ttnn.to_layout(attn, layout=ttnn.TILE_LAYOUT)
            attn = ttnn.to_device(attn, device=self.device)

        attn = ttnn.softmax(attn, dim=-1)

        x = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )
        x = ttnn.permute(x, (0, 2, 1, 3))

        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.from_device(x)

        x = ttnn.reshape(x, (x.shape[0], x.shape[1], C))
        x = ttnn.to_device(x, device=self.device)
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)

        proj_weight = self.parameters.proj.weight
        proj_bias = self.parameters.proj.bias

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        x = ttnn.linear(
            x,
            proj_weight,
            bias=proj_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )

        # Torch is used since ttnn doesn't support reshape or permute of 6D tensor
        x = ttnn.to_torch(x)
        # reverse windows
        x = x.view(
            B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        x = ttnn.reshape(x, (B, pad_H, pad_W, C))

        # reverse cyclic shift
        # Torch is used since ttnn doesn't support roll, should check this
        if sum(self.shift_size) > 0:
            x = ttnn.to_torch(x)
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # unpad features
        x = x[:, :H, :W, :]

        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        return x
