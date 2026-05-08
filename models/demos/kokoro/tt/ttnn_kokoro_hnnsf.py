# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of hn-nsf source generation used by Kokoro ISTFTNet.

NOTE: This implementation is deterministic (no random init/noise) to avoid a hard
dependency on synchronized RNG across host/device. It is sufficient for bringing
up the remaining host fallbacks in Kokoro.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn


@dataclass(frozen=True)
class HnNsfSourceParams:
    sampling_rate: float
    harmonic_num: int
    sine_amp: float
    voiced_threshold: float
    linear_weight: ttnn.Tensor  # [out=1, in=dim]
    linear_bias: ttnn.Tensor  # [1,1,1,out]


def preprocess_hnnsf_source(torch_source_module: torch.nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16):
    # SourceModuleHnNSF has l_linear: Linear(dim,1) and l_tanh.
    w = torch_source_module.l_linear.weight.detach().cpu().to(torch.float32)  # [1, dim]
    b = torch_source_module.l_linear.bias.detach().cpu().to(torch.float32)  # [1]
    w_tt = ttnn.from_torch(
        w, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_tt = ttnn.from_torch(
        b.reshape(1, 1, 1, -1),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sg = torch_source_module.l_sin_gen
    return HnNsfSourceParams(
        sampling_rate=float(sg.sampling_rate),
        harmonic_num=int(sg.harmonic_num),
        sine_amp=float(sg.sine_amp),
        voiced_threshold=float(sg.voiced_threshold),
        linear_weight=w_tt,
        linear_bias=b_tt,
    )


def hnnsf_source(
    *,
    f0_btl: ttnn.Tensor,  # [B, T, 1] in Hz (already upsampled to waveform rate)
    params: HnNsfSourceParams,
    device: ttnn.Device,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Returns:
      - sine_merge_btl: [B, T, 1]
      - uv_btl:         [B, T, 1]
    """
    x = f0_btl
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    # uv = (f0 > threshold)
    uv = ttnn.gt(x, params.voiced_threshold)
    uv = ttnn.to_layout(uv, ttnn.ROW_MAJOR_LAYOUT)
    uv_f = ttnn.where(uv, 1.0, 0.0)

    # fn = f0 * [1..harmonic_num+1] -> [B,T,dim]
    dim = params.harmonic_num + 1
    mult = torch.arange(1, dim + 1, dtype=torch.float32).reshape(1, 1, dim)
    mult_tt = ttnn.from_torch(mult, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    fn = ttnn.multiply(x, mult_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # phase = cumsum((fn/sr)%1, dim=1) * 2pi
    rad = ttnn.multiply(fn, 1.0 / params.sampling_rate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rad = ttnn.remainder(rad, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    phase = ttnn.cumsum(rad, dim=1)
    phase = ttnn.multiply(phase, 2.0 * 3.141592653589793, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sines = ttnn.sin(phase)

    # Apply amp and uv (no noise)
    sines = ttnn.multiply(sines, params.sine_amp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    uv_rep = ttnn.repeat(uv_f, (1, 1, dim))
    sines = ttnn.multiply(sines, uv_rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # linear + tanh: [B,T,dim] -> [B,T,1]
    # ttnn.linear expects [*,in] with weight [out,in] when transpose_b=True.
    sines_2d = ttnn.reshape(sines, (sines.shape[0] * sines.shape[1], dim), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sines_2d = ttnn.to_layout(sines_2d, ttnn.TILE_LAYOUT)
    y = ttnn.linear(
        sines_2d, params.linear_weight, bias=params.linear_bias, transpose_b=True, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y = ttnn.tanh(y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = ttnn.reshape(y, (f0_btl.shape[0], f0_btl.shape[1], 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return y, uv_f
