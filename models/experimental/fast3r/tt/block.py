"""tt-nn building blocks for Fast3R encoder/decoder.

Each module owns its weights on the device and is callable on a (B, 1, N, C) device tensor.
"""
from __future__ import annotations

import torch
import ttnn

from .mlp import TtMlp, to_device_bias, to_device_weight


class TtLayerNorm:
    """eps matches reference.model (1e-6)."""

    def __init__(self, device, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6):
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.bias = ttnn.from_torch(
            bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.eps = eps

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)


class TtNormMlp:
    """norm2 + mlp fragment of an encoder/decoder block — used to verify the residual-free path."""

    def __init__(self, device, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b, *, eps: float = 1e-6):
        self.norm = TtLayerNorm(device, norm_w, norm_b, eps)
        self.mlp = TtMlp(device, fc1_w, fc1_b, fc2_w, fc2_b)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.mlp(self.norm(x))
