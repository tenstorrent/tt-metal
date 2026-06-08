# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro ``AdaIN1d`` from ``reference/istftnet.py``.

``AdaIN1d`` applies ``InstanceNorm1d`` (with optional affine) along the time axis, then scales and
shifts per channel using ``(1 + gamma(s)) * y + beta(s)`` from a linear map of the style vector.

Activations use **NLC** layout ``[B, L, C]`` (length ``L``, channels ``C``). PyTorch appears only in
the ``preprocess_tt_*`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

import ttnn

from .tt_matmul_memory import maybe_reshard_to_caller, style_linear_plan


@dataclass(frozen=True)
class TTInstanceNorm1dParams:
    """Affine parameters for instance norm over ``L`` in ``[B, L, C]`` (``None`` = identity scale/shift)."""

    weight: Optional[ttnn.Tensor]
    bias: Optional[ttnn.Tensor]
    eps: float


def tt_instance_norm_1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTInstanceNorm1dParams,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    Instance norm over the length dimension (dim ``1``) for each batch row and channel.

    Matches ``nn.InstanceNorm1d`` on inputs ``[B, C, L]`` after conversion to NLC.
    """
    mean = ttnn.mean(x_nlc, dim=1, keepdim=True, memory_config=memory_config)
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
class TTAdaIN1dParams:
    """Device weights for :class:`TTAdaIN1d`."""

    fc_weight: ttnn.Tensor
    fc_bias: ttnn.Tensor
    instancenorm: TTInstanceNorm1dParams
    num_features: int


def preprocess_tt_instance_norm_1d(
    inn: nn.InstanceNorm1d,
    device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTInstanceNorm1dParams:
    """Upload ``nn.InstanceNorm1d`` affine parameters for :func:`tt_instance_norm_1d_nlc`."""
    eps = float(inn.eps)
    if inn.affine and inn.weight is not None and inn.bias is not None:
        inf_w = ttnn.from_torch(
            inn.weight.detach().cpu().reshape(1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inf_b = ttnn.from_torch(
            inn.bias.detach().cpu().reshape(1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        inf_w = None
        inf_b = None
    return TTInstanceNorm1dParams(weight=inf_w, bias=inf_b, eps=eps)


def preprocess_tt_adain_1d(
    module: nn.Module,
    device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTAdaIN1dParams:
    """Upload PyTorch ``AdaIN1d`` (``.norm`` + ``.fc``) for :class:`TTAdaIN1d`."""
    fc = module.fc
    inn: nn.InstanceNorm1d = module.norm
    c = int(inn.num_features)

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

    inst = preprocess_tt_instance_norm_1d(inn, device, weights_dtype=weights_dtype)

    return TTAdaIN1dParams(
        fc_weight=fc_w,
        fc_bias=fc_b,
        instancenorm=inst,
        num_features=c,
    )


class TTAdaIN1d:
    """Adaptive instance norm: ``(1 + gamma(s)) * InstanceNorm(x) + beta(s)`` on NLC tensors."""

    __slots__ = ("params",)

    def __init__(self, params: TTAdaIN1dParams) -> None:
        self.params = params

    def forward(
        self,
        x_nlc: ttnn.Tensor,
        style_bs: ttnn.Tensor,
        *,
        compute_kernel_config,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        p = self.params
        c = p.num_features

        y = tt_instance_norm_1d_nlc(x_nlc=x_nlc, params=p.instancenorm, memory_config=memory_config)

        b = int(style_bs.shape[0])
        style_out_mc, style_reshard = style_linear_plan(b, int(p.fc_weight.shape[-1]), 2 * c)
        h = ttnn.linear(
            style_bs,
            p.fc_weight,
            bias=p.fc_bias,
            transpose_b=True,
            memory_config=style_out_mc if style_out_mc is not None else memory_config,
            compute_kernel_config=compute_kernel_config,
        )
        if style_reshard:
            h = maybe_reshard_to_caller(h, memory_config)
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
