# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.swin_v2.tt.utils import ttnn_custom_normalize


class TtShiftedWindowAttentionV2:
    def __init__(
        self,
        parameters,
        device,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        attn_mask=None,
    ):
        self.parameters = parameters
        self.device = device
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attn_mask = attn_mask

    def forward(self, x):
        relative_position_bias = self.parameters["relative_position_bias"]
        logit_scale = self.parameters["logit_scale"]

        B, H, W, C = x.shape
        if self.window_size[0] >= H:
            self.shift_size[0] = 0
        if self.window_size[1] >= W:
            self.shift_size[1] = 0

        num_windows = (H // self.window_size[0]) * (W // self.window_size[1])

        x = ttnn.reshape(
            x,
            (
                (
                    B,
                    H // self.window_size[0],
                    self.window_size[0],
                    W // self.window_size[1],
                    self.window_size[1],
                    C,
                )
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5), memory_config=ttnn.L1_MEMORY_CONFIG)

        qkv_bias = self.parameters.qkv.bias

        x = ttnn.reshape(
            x, (B * num_windows, self.window_size[0] * self.window_size[1], C), memory_config=ttnn.L1_MEMORY_CONFIG
        )

        qkv_weight = self.parameters.qkv.weight

        qkv = ttnn.linear(
            x,
            qkv_weight,
            bias=qkv_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.reshape(
            qkv, (x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads), memory_config=ttnn.L1_MEMORY_CONFIG
        )

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

        k = ttnn_custom_normalize(k, dim=-1, device=self.device)
        k = ttnn.permute(k, [0, 1, 3, 2])
        q = ttnn_custom_normalize(q, dim=-1, device=self.device)
        attn = q @ k

        logit_scale = ttnn.clamp(logit_scale, max=4.605170185988092, memory_config=ttnn.L1_MEMORY_CONFIG)

        logit_scale = ttnn.exp(logit_scale, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn = ttnn.multiply(attn, logit_scale, use_legacy=False)

        ttnn.deallocate(qkv)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        attn = ttnn.add(attn, relative_position_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        if sum(self.shift_size) > 0:
            attn = ttnn.add(attn, self.attn_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (-1, self.num_heads, x.shape[1], x.shape[1]), memory_config=ttnn.L1_MEMORY_CONFIG)

        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(v)
        ttnn.deallocate(attn)
        x = ttnn.permute(x, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.reshape(x, (x.shape[0], x.shape[1], C), memory_config=ttnn.L1_MEMORY_CONFIG)

        proj_weight = self.parameters.proj.weight
        proj_bias = self.parameters.proj.bias

        x = ttnn.linear(
            x,
            proj_weight,
            bias=proj_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        x = ttnn.reshape(
            x,
            (
                B,
                H // self.window_size[0],
                W // self.window_size[1],
                self.window_size[0],
                self.window_size[1],
                C,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5), memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.reshape(x, (B, H, W, C), memory_config=ttnn.L1_MEMORY_CONFIG)

        x = x[:, :H, :W, :]
        ttnn.DumpDeviceProfiler(self.device)
        return x
