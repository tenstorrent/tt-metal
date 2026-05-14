# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.modules.AdaLayerNorm`.

Reference math, with ``x`` in ``[B, T, C]`` and ``s`` in ``[B, style_dim]``:

    h = fc(s)                          # [B, 2C]
    gamma, beta = chunk(h, 2)          # each [B, C], broadcast over T
    y = LayerNorm_affine_free(x, eps)
    out = (1 + gamma) * y + beta

PyTorch is used only at preprocessing time to read ``fc`` weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

import ttnn


@dataclass(frozen=True)
class TTAdaLayerNormParams:
    """Device-resident weights for :class:`TTAdaLayerNorm`."""

    fc_weight: ttnn.Tensor
    fc_bias: ttnn.Tensor
    channels: int
    eps: float


def preprocess_tt_ada_layer_norm(
    module: nn.Module,
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTAdaLayerNormParams:
    """Upload ``AdaLayerNorm.fc`` (``nn.Linear(style_dim, 2C)``) to the device."""
    fc = module.fc
    fc_w = ttnn.from_torch(
        fc.weight.detach().cpu(),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    fc_b = ttnn.from_torch(
        fc.bias.detach().cpu().reshape(1, 1, 1, -1),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return TTAdaLayerNormParams(
        fc_weight=fc_w,
        fc_bias=fc_b,
        channels=int(module.channels),
        eps=float(module.eps),
    )


class TTAdaLayerNorm:
    """Adaptive LayerNorm: ``LN(x) * (1 + gamma(s)) + beta(s)`` over the channel (last) dim."""

    __slots__ = ("params",)

    def __init__(self, params: TTAdaLayerNormParams) -> None:
        self.params = params

    def forward(
        self,
        x_nlc: ttnn.Tensor,
        style_bc: ttnn.Tensor,
        *,
        compute_kernel_config,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """
        Args:
            x_nlc: ``[B, T, C]`` on device (tile layout).
            style_bc: ``[B, style_dim]`` on device (tile layout).

        Returns:
            ``[B, T, C]`` on device (tile layout).
        """
        p = self.params
        c = p.channels

        y = ttnn.layer_norm(
            x_nlc,
            epsilon=p.eps,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )

        h = ttnn.linear(
            style_bc,
            p.fc_weight,
            bias=p.fc_bias,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )
        # ``ttnn.linear`` on a rank-2 input may emit leading singleton dims; collapse them.
        while len(h.shape) > 2:
            h = ttnn.squeeze(h, 0)
        b = int(h.shape[0])
        h = ttnn.reshape(h, [b, 1, 2 * c], memory_config=memory_config)

        gamma, beta = ttnn.chunk(h, 2, dim=-1)
        ttnn.deallocate(h)

        scale = ttnn.add(gamma, 1.0, memory_config=memory_config)
        ttnn.deallocate(gamma)
        y = ttnn.multiply(y, scale, memory_config=memory_config)
        ttnn.deallocate(scale)
        out = ttnn.add(y, beta, memory_config=memory_config)
        ttnn.deallocate(y)
        ttnn.deallocate(beta)
        return out

    __call__ = forward
