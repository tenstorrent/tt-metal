# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro TTNN convolution helpers.

- `weight_norm_weight`: compute effective Conv1d/ConvTranspose1d weight from (g, v)
- `conv1d_nlc`: TTNN conv1d wrapper for activations in NLC layout
- `conv_transpose1d_nlc`: map 1D transpose-conv onto TTNN conv_transpose2d using H=1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn


def weight_norm_weight(weight_v: torch.Tensor, weight_g: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute effective weight for PyTorch `torch.nn.utils.weight_norm`.

    For Conv1d, v has shape [out_ch, in_ch/groups, k].
    g has shape [out_ch, 1, 1] or [out_ch, 1] depending on module.
    """
    v = weight_v
    g = weight_g
    while g.dim() < v.dim():
        g = g.unsqueeze(-1)
    v_norm = torch.linalg.vector_norm(v.reshape(v.shape[0], -1), ord=2, dim=1).clamp_min(eps)
    v_norm = v_norm.view(v.shape[0], *([1] * (v.dim() - 1)))
    return v * (g / v_norm)


@dataclass(frozen=True)
class Conv1dParams:
    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor]
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    groups: int = 1


def conv1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: Conv1dParams,
    device,
    conv_config: Optional[ttnn.Conv1dConfig] = None,
    compute_config=None,
    out_dtype=ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    Conv1d wrapper with activations in [B, L, C] (NLC), matching TTNN conv1d tests.

    Returns TTNN tensor [B, out_L, out_C] (NLC).
    """
    if conv_config is None:
        conv_config = ttnn.Conv1dConfig(weights_dtype=params.weight.dtype)
        # Keep conv config tensors in DRAM to avoid small-L1 allocations on some setups.
        conv_config.config_tensors_in_dram = True
        conv_config.deallocate_activation = True
        # Reduce L1 circular buffer pressure for large 1D convs (generator stack).
        # Keep this conditional to avoid destabilizing small convs.
        if params.out_channels >= 256 or params.kernel_size >= 7:
            try:
                conv_config.force_split_reader = True
            except Exception:
                pass
    if compute_config is None:
        # Wormhole: prefer HiFi3 when using fp32 accumulation (HiFi4 can be worse due to HW bug).
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
        batch_size=x_nlc.shape[0],
        input_length=x_nlc.shape[1],
        conv_config=conv_config,
        compute_config=compute_config,
        groups=params.groups,
        dtype=out_dtype,
        return_output_dim=True,
    )
    # conv1d returns a device tensor; normalize to [B, out_len, out_C] on device
    if len(y.shape) == 4 and y.shape[0] == 1:
        y = ttnn.squeeze(y, 0)
    if len(y.shape) == 4 and y.shape[1] == 1:
        # [B, 1, L, C] -> [B, L, C]
        y = ttnn.reshape(y, [y.shape[0], y.shape[2], y.shape[3]], memory_config=memory_config)
    if len(y.shape) == 3 and y.shape[1] != out_len:
        y = ttnn.reshape(y, (x_nlc.shape[0], out_len, y.shape[-1]), memory_config=memory_config)
    return y


@dataclass(frozen=True)
class ConvTranspose1dParams:
    # Map to TTNN conv_transpose2d weights: [in_ch, out_ch/groups, 1, k]
    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor]
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    output_padding: int
    groups: int = 1


def conv_transpose1d_nlc(
    *,
    x_nlc: ttnn.Tensor,
    params: ConvTranspose1dParams,
    device,
    conv_config: Optional[ttnn.Conv2dConfig] = None,
    compute_config=None,
    out_dtype=ttnn.bfloat16,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    1D transpose convolution using TTNN conv_transpose2d by treating length as width and H=1.

    Input:  [B, L, Cin] (NLC)
    Output: [B, out_L, Cout] (NLC)
    """
    # conv_transpose2d reshapes internally; avoid tile-padding volume mismatches by using row-major activations here
    if x_nlc.layout != ttnn.ROW_MAJOR_LAYOUT:
        x_nlc = ttnn.to_layout(x_nlc, ttnn.ROW_MAJOR_LAYOUT)
    # NLC -> NHWC with H=1, W=L
    x = ttnn.reshape(x_nlc, (x_nlc.shape[0], 1, x_nlc.shape[1], x_nlc.shape[2]), memory_config=memory_config)
    if conv_config is None:
        conv_config = ttnn.Conv2dConfig(weights_dtype=params.weight.dtype)
    if compute_config is None:
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True
        )

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
        batch_size=x_nlc.shape[0],
        input_height=1,
        input_width=x_nlc.shape[1],
        conv_config=conv_config,
        compute_config=compute_config,
        groups=params.groups,
        dtype=out_dtype,
        return_output_dim=True,
    )
    # y is NHWC: [B, 1, out_L, Cout] -> [B, out_L, Cout]
    y = ttnn.reshape(y, (y.shape[0], out_hw[1], y.shape[3]), memory_config=memory_config)
    return y
