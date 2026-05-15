# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro TTNN convolution helpers (TT-prefixed params, ``tt_`` functions).

- :func:`tt_weight_norm_materialize` — effective Conv weight from ``(v, g)`` tensors
- :func:`tt_conv1d_nlc` — Conv1d on NLC activations ``[B, L, C]``
- :func:`tt_conv_transpose1d_nlc` — ConvTranspose1d via ``conv_transpose2d`` (``spatial_style`` width vs height)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn


def tt_weight_norm_materialize(weight_v: torch.Tensor, weight_g: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Effective weight for legacy ``torch.nn.utils.weight_norm`` (``v``, ``g`` tensors).

    For Conv1d, ``v`` has shape ``[out_ch, in_ch/groups, k]``; ``g`` is broadcastable to ``v``.
    """
    v = weight_v
    g = weight_g
    while g.dim() < v.dim():
        g = g.unsqueeze(-1)
    v_norm = torch.linalg.vector_norm(v.reshape(v.shape[0], -1), ord=2, dim=1).clamp_min(eps)
    v_norm = v_norm.view(v.shape[0], *([1] * (v.dim() - 1)))
    return v * (g / v_norm)


@dataclass(frozen=True)
class TTConv1dParams:
    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor]
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    groups: int = 1
    dilation: int = 1


def tt_conv1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTConv1dParams,
    device,
    conv_config: Optional[ttnn.Conv1dConfig] = None,
    compute_config=None,
    out_dtype=ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    Conv1d on activations ``[B, L, C]`` (NLC). Returns ``[B, out_L, out_C]`` (NLC).
    """
    if conv_config is None:
        conv_config = ttnn.Conv1dConfig(weights_dtype=params.weight.dtype)
        conv_config.config_tensors_in_dram = True
        conv_config.deallocate_activation = True
        if params.out_channels >= 256 or params.kernel_size >= 7:
            try:
                conv_config.force_split_reader = True
            except Exception:
                pass
    if compute_config is None:
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True
        )

    y, out_len = ttnn.conv1d(
        input_tensor=x_nlc,
        weight_tensor=params.weight,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        device=device,
        bias_tensor=params.bias,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=params.padding,
        dilation=params.dilation,
        batch_size=x_nlc.shape[0],
        input_length=x_nlc.shape[1],
        conv_config=conv_config,
        compute_config=compute_config,
        groups=params.groups,
        dtype=out_dtype,
        return_output_dim=True,
    )
    if len(y.shape) == 4 and y.shape[0] == 1:
        y = ttnn.squeeze(y, 0)
    if len(y.shape) == 4 and y.shape[1] == 1:
        y = ttnn.reshape(y, [y.shape[0], y.shape[2], y.shape[3]], memory_config=memory_config)
    if len(y.shape) == 3 and y.shape[1] != out_len:
        y = ttnn.reshape(y, (x_nlc.shape[0], out_len, y.shape[-1]), memory_config=memory_config)
    return y


@dataclass(frozen=True)
class TTConvTranspose1dParams:
    # TTNN conv_transpose2d weights (NHWC op): default ``[in_ch, out_ch/groups, 1, k]`` (``spatial_style="width"``);
    # Kokoro istftnet depthwise pool uses ``[in_ch, out_ch/groups, k, 1]`` (``spatial_style="height"``).
    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor]
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    output_padding: int
    groups: int = 1
    mirror_kernel: bool = True
    spatial_style: str = "width"  # "width" | "height"


def tt_conv_transpose1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: TTConvTranspose1dParams,
    device,
    conv_config: Optional[ttnn.Conv2dConfig] = None,
    compute_config=None,
    out_dtype=ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """1D transpose conv via ``conv_transpose2d`` (NHWC ``[N,H,W,C]`` staging inside the op)."""
    if x_nlc.layout != ttnn.ROW_MAJOR_LAYOUT:
        x_nlc = ttnn.to_layout(x_nlc, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x_nlc, (x_nlc.shape[0], 1, x_nlc.shape[1], x_nlc.shape[2]), memory_config=memory_config)
    if conv_config is None:
        conv_config = ttnn.Conv2dConfig(weights_dtype=params.weight.dtype)
    if compute_config is None:
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True
        )

    bsz = int(x_nlc.shape[0])
    seq = int(x_nlc.shape[1])
    if params.spatial_style == "height":
        # Matches ``ttnn_adain_resblk_encode._TtDepthwiseConvTransposePool`` (Kokoro istftnet pool).
        y, out_hw = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=params.weight,
            in_channels=params.in_channels,
            out_channels=params.out_channels,
            device=device,
            bias_tensor=params.bias,
            kernel_size=(params.kernel_size, 1),
            stride=(params.stride, 1),
            padding=(params.padding, 0),
            output_padding=(params.output_padding, 0),
            dilation=(1, 1),
            batch_size=bsz,
            input_height=seq,
            input_width=1,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=params.groups,
            dtype=out_dtype,
            mirror_kernel=params.mirror_kernel,
            return_output_dim=True,
        )
        oh, ow = int(out_hw[0]), int(out_hw[1])
        flat = oh * ow
        y = ttnn.reshape(y, (y.shape[0], flat, y.shape[3]), memory_config=memory_config)
        return y

    y, out_hw = ttnn.conv_transpose2d(
        input_tensor=x,
        weight_tensor=params.weight,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        device=device,
        bias_tensor=params.bias,
        kernel_size=(1, params.kernel_size),
        stride=(1, params.stride),
        padding=(0, params.padding),
        output_padding=(0, params.output_padding),
        dilation=(1, 1),
        batch_size=bsz,
        input_height=1,
        input_width=seq,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=params.groups,
        dtype=out_dtype,
        mirror_kernel=params.mirror_kernel,
        return_output_dim=True,
    )
    y = ttnn.reshape(y, (y.shape[0], out_hw[1], y.shape[3]), memory_config=memory_config)
    return y
