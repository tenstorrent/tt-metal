# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm over the last dimension.

Qwen3_5RMSNorm is ZERO-CENTERED because of training stability

    output = x * rsqrt(mean(x^2) + eps) * (1 + weight)

   The stored weight is centred on zero (initialized to zeros), hence the `1 +`.

   The cheap way to get the `1 +` is to add 1.0 to the weight ONCE at load time;
   then a plain ttnn.rms_norm is exact, with no extra runtime op.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

EPS = 1e-6  # rms_norm_eps


class TtQwen36RmsNorm(LightweightModule):
    """
    Zero-centered RMSNorm: one fused ttnn.rms_norm against a pre-offset weight.

    op computes x * rsqrt(mean(x^2) + eps) * weight.

    Shapes:
        x        [..., dim]     TILE layout
        weight   [dim]
        output   [..., dim]     same shape as input
    """

    def __init__(self, device, torch_weight: torch.Tensor, eps: float = EPS):
        self.device = device
        self.eps = eps

        # Fold the `1 +` into the weight ONCE, on host, so the runtime op is a
        # plain fused rms_norm with no extra add. Done in fp32 so the addition
        # itself is exact; the result still rounds to bf16 on the way to device.
        self.weight = ttnn.as_tensor(
            torch_weight.float() + 1.0,  # [dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[..., dim] -> [..., dim]."""
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)
