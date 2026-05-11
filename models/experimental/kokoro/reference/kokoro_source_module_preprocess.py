# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side preprocessing for :class:`models.experimental.kokoro.tt.ttnn_source_module_hn_nsf.SourceModuleHnNSF`.

Weights and resampling matrices are built with PyTorch here so the TTNN module stays
torch-free at import and inference time.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

import ttnn


def _linear_interp_matrix_for_scale(t_in: int, scale_factor: float, *, align_corners: bool = False) -> torch.Tensor:
    """Return G (t_out, t_in) with y = G @ x for column x, matching ``F.interpolate`` 1D linear."""
    t_out = F.interpolate(
        torch.zeros(1, 1, t_in, dtype=torch.float64),
        scale_factor=scale_factor,
        mode="linear",
        align_corners=align_corners,
    ).shape[-1]
    g = torch.zeros(t_out, t_in, dtype=torch.float64)
    for i in range(t_in):
        e = torch.zeros(1, 1, t_in, dtype=torch.float64)
        e[0, 0, i] = 1.0
        g[:, i] = F.interpolate(e, scale_factor=scale_factor, mode="linear", align_corners=align_corners).view(-1)
    return g.float()


def preprocess_source_module_hn_nsf_parameters(
    torch_m: Any,
    device,
    time_len: int,
    *,
    align_corners: bool = False,
) -> dict:
    """
    Extract weights and precompute 1D linear resampling matrices for a fixed ``time_len``
    (F0 length after ``f0_upsamp``, shape ``(batch, time_len, 1)``).
    """
    sg = torch_m.l_sin_gen
    upsample_scale = float(sg.upsample_scale)
    if upsample_scale <= 0:
        raise ValueError("upsample_scale must be positive")

    g_down = _linear_interp_matrix_for_scale(time_len, 1.0 / upsample_scale, align_corners=align_corners)
    t_down = int(g_down.shape[0])
    g_up = _linear_interp_matrix_for_scale(t_down, upsample_scale, align_corners=align_corners)
    if int(g_up.shape[0]) != time_len:
        raise RuntimeError(
            f"Interp size mismatch: time_len={time_len}, t_down={t_down}, "
            f"g_up rows={g_up.shape[0]} (expected {time_len})"
        )

    w = torch_m.l_linear.weight.data.T.contiguous().to(dtype=torch.float32)
    b = torch_m.l_linear.bias.data.to(dtype=torch.float32)
    harmonics = torch.arange(1, sg.harmonic_num + 2, dtype=torch.float32).view(1, 1, -1)
    harmonic_rand_mask = torch.ones(1, 1, harmonics.shape[-1], dtype=torch.float32)
    harmonic_rand_mask[0, 0, 0] = 0.0

    dram = ttnn.DRAM_MEMORY_CONFIG
    l1 = ttnn.L1_MEMORY_CONFIG

    return {
        "time_len": int(time_len),
        "t_down": int(t_down),
        "upsample_scale": upsample_scale,
        "sampling_rate": float(sg.sampling_rate),
        "harmonic_num": int(sg.harmonic_num),
        "sine_amp": float(torch_m.sine_amp),
        "noise_std": float(sg.noise_std),
        "voiced_threshold": float(sg.voiced_threshold),
        "linear_weight": ttnn.from_torch(
            w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "linear_bias": ttnn.from_torch(
            b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "mat_down": ttnn.from_torch(
            g_down.T.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "mat_up": ttnn.from_torch(
            g_up.T.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "harmonics": ttnn.from_torch(
            harmonics, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "harmonic_rand_mask": ttnn.from_torch(
            harmonic_rand_mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "two_pi": ttnn.from_torch(
            torch.tensor([[[2.0 * math.pi]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        ),
        "inv_sampling_rate": ttnn.from_torch(
            torch.tensor([[[1.0 / float(sg.sampling_rate)]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        ),
        "one": ttnn.from_torch(
            torch.tensor([[[1.0]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        ),
        "l1_cfg": l1,
    }
