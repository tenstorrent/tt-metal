# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn


class FinalConv:
    "Final Conv in the form of ttnn.matmul since kernel size is 1"

    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def load_state_dict(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        weight_prefix = f"{module_prefix}.weight" if module_prefix else "weight"
        bias_prefix = f"{module_prefix}.bias" if module_prefix else "bias"
        self.weight = ttnn.from_torch(
            params_dict[weight_prefix].permute(1, 0, 2, 3, 4),
            dtype=ttnn.bfloat16,
            device=device,
        )
        self.weight = ttnn.reshape(
            self.weight,
            (self.in_channels, self.out_channels),
        )
        out_padding = (32 - self.out_channels % 32) % 32
        if out_padding > 0:
            self.weight = ttnn.pad(
                self.weight,
                [(0, 0), (0, out_padding)],
                0,
            )
        self.bias = ttnn.from_torch(
            params_dict[bias_prefix],
            dtype=ttnn.bfloat16,
            device=device,
        )
        self.bias = ttnn.reshape(self.bias, (self.out_channels,))
        if out_padding > 0:
            self.bias = ttnn.pad(
                self.bias,
                [(0, out_padding)],
                0,
            )

        self.weight = ttnn.to_layout(self.weight, ttnn.TILE_LAYOUT)

    def __call__(self, x, device) -> ttnn.Tensor:
        N, D, H, W, C = x.shape
        x0 = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if x.layout != ttnn.TILE_LAYOUT:
            ttnn.deallocate(x)
        x1 = ttnn.reshape(x0, (N * D * H * W, C))

        x2 = ttnn.linear(
            x1,
            self.weight,
            bias=self.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            activation="sigmoid",
        )

        ttnn.deallocate(x1)
        x3 = ttnn.slice(x2, [0, 0], [N * D * H * W, self.out_channels])
        if x3.buffer_address() != x2.buffer_address():
            ttnn.deallocate(x2)
        return ttnn.reshape(x3, (N, D, H, W, self.out_channels))
