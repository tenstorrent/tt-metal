# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn


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

    def forward(self, input_tensor):
        relative_position_bias = self.parameters[
            "relative_position_bias"
        ]  # relative position bias is taken from torch since it won't differ from input

        B, H, W, C = input_tensor.shape
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_values = (B, H + pad_r, W + pad_b, C)
        input_tensor = ttnn.pad(input_tensor, pad_values, [0, 0, 0, 0], 0)
        _, pad_H, pad_W, _ = input_tensor.shape

        self.shift_size = self.shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        # cyclic shift
        if sum(self.shift_size) > 0:
            input_tensor = ttnn.roll(input_tensor, [-self.shift_size[0], -self.shift_size[1]], [1, 2])

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])

        input_tensor = ttnn.reshape(
            input_tensor,
            (
                (
                    B,
                    pad_H // self.window_size[0],
                    self.window_size[0],
                    pad_W // self.window_size[1],
                    self.window_size[1],
                    C,
                )
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        input_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2, 4, 5), memory_config=ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.reshape(input_tensor, (B * num_windows, self.window_size[0] * self.window_size[1], C))

        qkv_weight = self.parameters.qkv.weight
        qkv_bias = self.parameters.qkv.bias

        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.linear(
            input_tensor,
            qkv_weight,
            bias=qkv_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.reshape(qkv, (input_tensor.shape[0], input_tensor.shape[1], 3, self.num_heads, C // self.num_heads))

        qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4), memory_config=ttnn.L1_MEMORY_CONFIG)

        q = qkv[0:1, :, :, :, :]
        k = qkv[1:2, :, :, :, :]
        v = qkv[2:3, :, :, :, :]
        q = ttnn.squeeze(q, 0)
        k = ttnn.squeeze(k, 0)
        v = ttnn.squeeze(v, 0)
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        q = q * (C // self.num_heads) ** -0.5
        k = ttnn.permute(k, (0, 1, 3, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        # add relative position bias
        attn = ttnn.add(attn, relative_position_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        if sum(self.shift_size) > 0:
            attn = attn + self.attn_mask
            attn = ttnn.to_layout(attn, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (-1, self.num_heads, input_tensor.shape[1], input_tensor.shape[1]))
            attn = ttnn.to_layout(attn, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(v)
        ttnn.deallocate(attn)
        input_tensor = ttnn.permute(input_tensor, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.reshape(input_tensor, (input_tensor.shape[0], input_tensor.shape[1], C))
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        proj_weight = self.parameters.proj.weight
        proj_bias = self.parameters.proj.bias

        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.linear(
            input_tensor,
            proj_weight,
            bias=proj_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # reverse windows
        output_tensor = ttnn.reshape(
            output_tensor,
            (
                B,
                pad_H // self.window_size[0],
                pad_W // self.window_size[1],
                self.window_size[0],
                self.window_size[1],
                C,
            ),
        )
        output_tensor = ttnn.permute(output_tensor, (0, 1, 3, 2, 4, 5), memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.reshape(output_tensor, (B, pad_H, pad_W, C))

        # reverse cyclic shift
        if sum(self.shift_size) > 0:
            output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.roll(output_tensor, [self.shift_size[0], self.shift_size[1]], [1, 2])

        # unpad features
        output_tensor = output_tensor[:, :H, :W, :]

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        return output_tensor
