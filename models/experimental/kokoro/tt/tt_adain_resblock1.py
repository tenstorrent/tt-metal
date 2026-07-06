# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.istftnet.AdaINResBlock1`.

Per stage (3 stages, ``dilation`` = ``(1, 3, 5)`` by default):

    xt = adain1[i](x, s)
    xt = snake1d(xt, alpha1[i])
    xt = conv1[i](xt)        # dilation = dilation[i]
    xt = adain2[i](xt, s)
    xt = snake1d(xt, alpha2[i])
    xt = conv2[i](xt)        # dilation = 1
    x  = xt + x

All activations stay in **NLC** ``[B, L, C]``. The reference's ``alpha`` parameters are
``[1, C, 1]`` (BCT broadcast); they are uploaded as ``[1, 1, C]`` here so the same parameter
broadcasts cleanly over ``[B, L, C]``.

PyTorch only appears in :func:`preprocess_tt_adain_resblock1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn
from torch.nn.utils import parametrize

import ttnn

from .tt_adain_1d import TTAdaIN1d, TTAdaIN1dParams, preprocess_tt_adain_1d
from .tt_conv import TTConv1dParams, tt_conv1d_nlc


@dataclass(frozen=True)
class TTAdaINResBlock1StageParams:
    """Weights for one of the three ``AdaINResBlock1`` stages."""

    adain1: TTAdaIN1dParams
    adain2: TTAdaIN1dParams
    conv1: TTConv1dParams
    conv2: TTConv1dParams
    alpha1: ttnn.Tensor  # [1, 1, C] (NLC broadcast)
    alpha2: ttnn.Tensor  # [1, 1, C]


@dataclass(frozen=True)
class TTAdaINResBlock1Params:
    """Device-resident weights for :class:`TTAdaINResBlock1`."""

    stages: tuple[TTAdaINResBlock1StageParams, ...]
    channels: int
    style_dim: int


def _strip_weight_norm(m: nn.Module) -> None:
    """Fold ``weight_norm`` into ``.weight`` (``parametrizations`` or legacy hook)."""
    if parametrize.is_parametrized(m, "weight"):
        parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        return
    try:
        nn.utils.remove_weight_norm(m, "weight")
    except ValueError:
        pass


def _conv1d_to_tt_params(conv: nn.Conv1d, device, *, weights_dtype) -> TTConv1dParams:
    """Upload a ``nn.Conv1d`` (after ``weight_norm`` is folded) for :func:`tt_conv1d_nlc`."""
    w = conv.weight.detach().cpu().unsqueeze(-1)  # [out, in, k, 1]
    w_tt = ttnn.from_torch(
        w,
        dtype=weights_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b_tt = None
    if conv.bias is not None:
        b_tt = ttnn.from_torch(
            conv.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
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
        dilation=int(conv.dilation[0]),
    )


def _alpha_to_tt(alpha_param, device, *, weights_dtype) -> ttnn.Tensor:
    """Reference ``alpha`` is ``[1, C, 1]`` (BCT broadcast); upload as ``[1, 1, C]`` for NLC."""
    a = alpha_param.detach().cpu().permute(0, 2, 1).contiguous()  # [1, 1, C]
    return ttnn.from_torch(
        a,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def preprocess_tt_adain_resblock1(
    module: nn.Module,
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
    conv_weights_dtype=ttnn.float32,
    alpha_dtype=ttnn.bfloat16,
) -> TTAdaINResBlock1Params:
    """Upload a reference ``AdaINResBlock1`` (3 stages × {2 AdaIN, 2 Conv, 2 alpha}) to device.

    Convolution weights stay at ``conv_weights_dtype`` (default fp32) to match PyTorch PCC
    on Wormhole — bf16 conv weights typically land near ~0.9 vs reference (same trade-off as
    :func:`preprocess_tt_adain_resblk_1d`).

    ``alpha`` (Snake1D) uses ``alpha_dtype`` (default bf16), decoupled from ``weights_dtype``: the
    generator runs Snake on bf16 activations, and a bf16 ``alpha`` keeps both per-sample multiplies
    in :func:`_tt_snake1d` on the fast pure-bf16 path instead of the ~6x-slower mixed bf16xfp32 op.
    """
    n_stages = len(module.convs1)
    assert (
        len(module.convs2) == n_stages
        and len(module.adain1) == n_stages
        and len(module.adain2) == n_stages
        and len(module.alpha1) == n_stages
        and len(module.alpha2) == n_stages
    ), "AdaINResBlock1 sub-lists must have matching length"

    stages: list[TTAdaINResBlock1StageParams] = []
    for i in range(n_stages):
        c1 = module.convs1[i]
        c2 = module.convs2[i]
        _strip_weight_norm(c1)
        _strip_weight_norm(c2)

        stage = TTAdaINResBlock1StageParams(
            adain1=preprocess_tt_adain_1d(module.adain1[i], device, weights_dtype=weights_dtype),
            adain2=preprocess_tt_adain_1d(module.adain2[i], device, weights_dtype=weights_dtype),
            conv1=_conv1d_to_tt_params(c1, device, weights_dtype=conv_weights_dtype),
            conv2=_conv1d_to_tt_params(c2, device, weights_dtype=conv_weights_dtype),
            alpha1=_alpha_to_tt(module.alpha1[i], device, weights_dtype=alpha_dtype),
            alpha2=_alpha_to_tt(module.alpha2[i], device, weights_dtype=alpha_dtype),
        )
        stages.append(stage)

    # ``channels`` comes from the (in == out) channel count of the first conv stage.
    channels = int(module.convs1[0].in_channels)
    style_dim = int(module.adain1[0].fc.in_features)

    return TTAdaINResBlock1Params(stages=tuple(stages), channels=channels, style_dim=style_dim)


def _tt_snake1d(
    x: ttnn.Tensor,
    alpha: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """``x + (1/alpha) * sin(alpha * x)^2`` (Snake1D from BigVGAN / Kokoro istftnet).

    ``alpha`` is a tiny per-channel ``[1, 1, C]`` weight. When its dtype does not match the
    activation we cast it once here so both full-length multiplies stay same-dtype: a mixed
    ``bf16 x fp32`` elementwise op is ~6x slower than a pure-``bf16`` one on Blackhole. ``alpha``
    is normally uploaded in the activation dtype (see :func:`preprocess_tt_adain_resblock1`), so
    this cast is a no-op on the common generator (bf16) path.
    """
    alpha_cast = None
    if alpha.dtype != x.dtype:
        alpha_cast = ttnn.typecast(alpha, x.dtype, memory_config=memory_config)
        alpha = alpha_cast
    ax = ttnn.multiply(x, alpha, memory_config=memory_config)
    sin_ax = ttnn.sin(ax, memory_config=memory_config)
    ttnn.deallocate(ax)
    sin_sq = ttnn.multiply(sin_ax, sin_ax, memory_config=memory_config)
    ttnn.deallocate(sin_ax)
    inv_alpha = ttnn.reciprocal(alpha, memory_config=memory_config)
    delta = ttnn.multiply(sin_sq, inv_alpha, memory_config=memory_config)
    ttnn.deallocate(sin_sq)
    ttnn.deallocate(inv_alpha)
    out = ttnn.add(x, delta, memory_config=memory_config)
    ttnn.deallocate(delta)
    if alpha_cast is not None:
        ttnn.deallocate(alpha_cast)
    return out


class TTAdaINResBlock1:
    """Three-stage AdaIN + Snake1D + dilated-Conv1d residual block (Kokoro istftnet)."""

    __slots__ = ("device", "params", "compute_kernel_config", "_adain1s", "_adain2s")

    def __init__(self, device: ttnn.Device, params: TTAdaINResBlock1Params) -> None:
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._adain1s: Sequence[TTAdaIN1d] = tuple(TTAdaIN1d(s.adain1) for s in params.stages)
        self._adain2s: Sequence[TTAdaIN1d] = tuple(TTAdaIN1d(s.adain2) for s in params.stages)

    def forward(
        self,
        x_nlc: ttnn.Tensor,
        style_bs: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """
        Args:
            x_nlc: ``[B, L, C]`` on device, TILE layout (``C == params.channels``).
            style_bs: ``[B, style_dim]`` on device, TILE layout.

        Returns:
            ``[B, L, C]`` on device, TILE layout.
        """
        ck = self.compute_kernel_config
        x = x_nlc

        for stage_params, ad1, ad2 in zip(self.params.stages, self._adain1s, self._adain2s):
            xt = ad1.forward(x, style_bs, compute_kernel_config=ck, memory_config=memory_config)
            xt = _tt_snake1d(xt, stage_params.alpha1, memory_config=memory_config)
            xt = tt_conv1d_nlc(
                x_nlc=xt,
                params=stage_params.conv1,
                device=self.device,
                compute_config=ck,
                memory_config=memory_config,
                preserve_input_dtype=True,
            )
            xt = ad2.forward(xt, style_bs, compute_kernel_config=ck, memory_config=memory_config)
            xt = _tt_snake1d(xt, stage_params.alpha2, memory_config=memory_config)
            xt = tt_conv1d_nlc(
                x_nlc=xt,
                params=stage_params.conv2,
                device=self.device,
                compute_config=ck,
                memory_config=memory_config,
                preserve_input_dtype=True,
            )
            x_new = ttnn.add(xt, x, memory_config=memory_config)
            ttnn.deallocate(xt)
            x = x_new

        return x

    __call__ = forward
