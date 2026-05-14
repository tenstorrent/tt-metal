# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro ``AdainResBlk1d`` from ``reference/istftnet.py``.

Uses **NLC** activations ``[B, L, C]`` internally. PyTorch is only used in
:func:`preprocess_tt_adain_resblk_1d` (weight upload + stripping ``weight_norm``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from torch.nn.utils import parametrize

import ttnn

from .tt_adain_1d import TTAdaIN1d, preprocess_tt_adain_1d
from .tt_conv import TTConv1dParams, TTConvTranspose1dParams, tt_conv1d_nlc, tt_conv_transpose1d_nlc
from .tt_upsample_1d import TTUpSample1d


@dataclass(frozen=True)
class TTAdainResBlk1dParams:
    """Preprocessed weights for :class:`TTAdainResBlk1d`."""

    layer_type: str
    learned_sc: bool
    norm1: TTAdaIN1dParams
    norm2: TTAdaIN1dParams
    conv1: TTConv1dParams
    conv2: TTConv1dParams
    conv1x1: Optional[TTConv1dParams]
    pool: Optional[TTConvTranspose1dParams]


def _strip_weight_norm_from_conv(m: nn.Module) -> None:
    """Fold ``weight_norm`` into plain ``.weight`` (``parametrizations`` or legacy hook)."""
    if parametrize.is_parametrized(m, "weight"):
        parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        return
    try:
        nn.utils.remove_weight_norm(m, "weight")
    except ValueError:
        pass


def _conv1d_to_tt_params(conv: nn.Conv1d, device, *, weights_dtype) -> TTConv1dParams:
    w = conv.weight.detach().cpu().unsqueeze(-1)
    b = conv.bias.detach().cpu() if conv.bias is not None else None
    w_tt = ttnn.from_torch(
        w,
        dtype=weights_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = (
        ttnn.from_torch(
            b.reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if b is not None
        else None
    )
    return TTConv1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=int(conv.in_channels),
        out_channels=int(conv.out_channels),
        kernel_size=int(conv.kernel_size[0]),
        stride=int(conv.stride[0]),
        padding=int(conv.padding[0]),
        groups=int(conv.groups),
    )


def _conv_transpose_pool_to_tt_params(m: nn.ConvTranspose1d, _device, *, weights_dtype) -> TTConvTranspose1dParams:
    """Depthwise upsample pool (``unsqueeze(-1)`` weight layout, **host** ROW_MAJOR like ``ttnn_adain_resblk_encode``).

    Keeping pool weights on **host** avoids the conv2d device path that requires pre-tilized weights
    (``Layout::TILE`` / folded matrix) before preparation.
    """
    w = m.weight.detach().cpu().unsqueeze(-1)
    w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = None
    if m.bias is not None:
        b_tt = ttnn.from_torch(
            m.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    return TTConvTranspose1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=int(m.in_channels),
        out_channels=int(m.out_channels),
        kernel_size=int(m.kernel_size[0]),
        stride=int(m.stride[0]),
        padding=int(m.padding[0]),
        output_padding=int(m.output_padding[0]),
        groups=int(m.groups),
        mirror_kernel=True,
        spatial_style="height",
    )


def preprocess_tt_adain_resblk_1d(
    module: nn.Module,
    device,
    *,
    weights_dtype=ttnn.bfloat16,
    conv_weights_dtype=ttnn.float32,
) -> TTAdainResBlk1dParams:
    """Upload ``AdainResBlk1d`` after materializing ``weight_norm`` weights.

    AdaIN MLP + instance-norm use ``weights_dtype`` (default bf16). Conv / conv-transpose weights and
    biases use ``conv_weights_dtype`` (default fp32) so ``ttnn.conv1d`` matmuls match PyTorch PCC
    (bf16-only conv weights typically land near ~0.9 vs reference).
    """
    for name in ("conv1", "conv2"):
        _strip_weight_norm_from_conv(getattr(module, name))
    if bool(getattr(module, "learned_sc", False)):
        _strip_weight_norm_from_conv(module.conv1x1)
    if not isinstance(module.pool, nn.Identity):
        _strip_weight_norm_from_conv(module.pool)

    learned_sc = bool(module.learned_sc)
    conv1x1 = _conv1d_to_tt_params(module.conv1x1, device, weights_dtype=conv_weights_dtype) if learned_sc else None

    pool_p: Optional[TTConvTranspose1dParams] = None
    if not isinstance(module.pool, nn.Identity):
        pool_p = _conv_transpose_pool_to_tt_params(module.pool, device, weights_dtype=conv_weights_dtype)

    layer_type = str(module.upsample.layer_type)

    return TTAdainResBlk1dParams(
        layer_type=layer_type,
        learned_sc=learned_sc,
        norm1=preprocess_tt_adain_1d(module.norm1, device, weights_dtype=weights_dtype),
        norm2=preprocess_tt_adain_1d(module.norm2, device, weights_dtype=weights_dtype),
        conv1=_conv1d_to_tt_params(module.conv1, device, weights_dtype=conv_weights_dtype),
        conv2=_conv1d_to_tt_params(module.conv2, device, weights_dtype=conv_weights_dtype),
        conv1x1=conv1x1,
        pool=pool_p,
    )


class TTAdainResBlk1d:
    """Residual AdaIN block: optional 2× nearest + depthwise transpose pool, two AdaIN + Conv stacks."""

    __slots__ = ("_compute_kernel_config", "_norm1", "_norm2", "_params", "_upsample", "device")

    def __init__(self, device, params: TTAdainResBlk1dParams) -> None:
        self.device = device
        self._params = params
        self._upsample = TTUpSample1d(params.layer_type)
        self._norm1 = TTAdaIN1d(params.norm1)
        self._norm2 = TTAdaIN1d(params.norm2)
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(
        self,
        x_nlc: ttnn.Tensor,
        style_bs: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        p = self._params
        ck = self._compute_kernel_config
        dev = self.device

        def shortcut(xi: ttnn.Tensor) -> ttnn.Tensor:
            sc = self._upsample.forward(xi, memory_config=memory_config)
            if p.learned_sc:
                assert p.conv1x1 is not None
                sc = tt_conv1d_nlc(
                    x_nlc=sc,
                    params=p.conv1x1,
                    device=dev,
                    compute_config=ck,
                    memory_config=memory_config,
                )
            return sc

        def residual(xi: ttnn.Tensor) -> ttnn.Tensor:
            x = self._norm1.forward(xi, style_bs, compute_kernel_config=ck, memory_config=memory_config)
            x = ttnn.leaky_relu(x, negative_slope=0.2, memory_config=memory_config)
            if p.pool is not None:
                x = tt_conv_transpose1d_nlc(
                    x_nlc=x,
                    params=p.pool,
                    device=dev,
                    compute_config=ck,
                    memory_config=memory_config,
                )
            x = tt_conv1d_nlc(x_nlc=x, params=p.conv1, device=dev, compute_config=ck, memory_config=memory_config)
            x = self._norm2.forward(x, style_bs, compute_kernel_config=ck, memory_config=memory_config)
            x = ttnn.leaky_relu(x, negative_slope=0.2, memory_config=memory_config)
            x = tt_conv1d_nlc(x_nlc=x, params=p.conv2, device=dev, compute_config=ck, memory_config=memory_config)
            return x

        r = residual(x_nlc)
        sc = shortcut(x_nlc)
        out = ttnn.add(r, sc, memory_config=memory_config)
        inv = 1.0 / math.sqrt(2.0)
        return ttnn.multiply(out, inv, memory_config=memory_config)

    __call__ = forward
