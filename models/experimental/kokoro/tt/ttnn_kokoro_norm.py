# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro TTNN normalization helpers (InstanceNorm1d + AdaIN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass(frozen=True)
class InstanceNorm1dParams:
    weight: Optional[ttnn.Tensor]  # gamma, shape [C] or broadcastable
    bias: Optional[ttnn.Tensor]  # beta
    eps: float = 1e-5


def instance_norm_1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: InstanceNorm1dParams,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    InstanceNorm over length dimension for NLC activations [B, L, C].

    Normalizes each (B,C) over L.
    """
    mean = ttnn.mean(x_nlc, dim=1, keepdim=True, memory_config=memory_config)  # [B,1,C]
    xc = ttnn.subtract(x_nlc, mean, memory_config=memory_config)
    var = ttnn.mean(ttnn.pow(xc, 2), dim=1, keepdim=True, memory_config=memory_config)
    inv_std = ttnn.rsqrt(ttnn.add(var, params.eps, memory_config=memory_config), memory_config=memory_config)
    y = ttnn.multiply(xc, inv_std, memory_config=memory_config)

    if params.weight is not None:
        y = ttnn.multiply(y, params.weight, memory_config=memory_config)
    if params.bias is not None:
        y = ttnn.add(y, params.bias, memory_config=memory_config)
    return y


@dataclass(frozen=True)
class AdaIN1dParams:
    # fc: style_dim -> 2*C
    fc_weight: ttnn.Tensor
    fc_bias: ttnn.Tensor
    instancenorm: InstanceNorm1dParams


def adain_1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    s_bc: ttnn.Tensor,  # [B, style_dim]
    params: AdaIN1dParams,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    AdaIN: (1+gamma(s))*IN(x) + beta(s), on NLC activations.

    `s_bc` is [B, style_dim] in TTNN.
    """
    y = instance_norm_1d_nlc(x_nlc=x_nlc, params=params.instancenorm, memory_config=memory_config)

    # fc(s) -> [B, 2C]
    h = ttnn.linear(
        s_bc,
        params.fc_weight,
        bias=params.fc_bias,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    # reshape to [B, 1, 2C] for broadcast over L
    h = ttnn.reshape(h, [h.shape[0], 1, h.shape[-1]], memory_config=memory_config)
    c2 = h.shape[-1]
    c = c2 // 2
    gamma = ttnn.slice(h, (0, 0, 0), (h.shape[0], 1, c), memory_config=memory_config)
    beta = ttnn.slice(h, (0, 0, c), (h.shape[0], 1, c2), memory_config=memory_config)

    one = ttnn.full(
        gamma.shape,
        fill_value=1.0,
        dtype=gamma.dtype,
        layout=gamma.layout,
        device=gamma.device(),
        memory_config=memory_config,
    )
    scale = ttnn.add(one, gamma, memory_config=memory_config)
    y = ttnn.multiply(y, scale, memory_config=memory_config)
    y = ttnn.add(y, beta, memory_config=memory_config)
    return y
