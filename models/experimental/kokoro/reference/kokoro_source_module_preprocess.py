# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side preprocessing for :class:`models.experimental.kokoro.tt.ttnn_source_module_hn_nsf.SourceModuleHnNSF`.

Builds TTNN device tensors for the SineGen downsampling / cumsum matrices and sparse linear-upsample tables
(``interp_up_*`` for gather-lerp on device). ``SineGen`` runs entirely on device (``KokoroTtnnSineGen``); the merge linear + tanh also run on TTNN.
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


def _interp_up_lerp_from_g_up(g_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse weights for 1D linear interpolation: each output uses at most two source bins.

    ``g_up`` is ``(t_out, t_in)`` with ``y = g_up @ x``. ``F.interpolate(mode='linear', align_corners=False)``
    uses a 2-tap kernel both directions, so rows have at most two nonzeros. Returning the gather indices and
    weights lets device ``ttnn.gather`` + mul/add reproduce ``F.interpolate`` exactly in fp32 — avoiding a
    wide bf16-quantised matmul (which drifts on long sequences and large phases).
    """
    t_out, _t_in = g_up.shape
    idx0 = torch.zeros(t_out, dtype=torch.int32)
    idx1 = torch.zeros(t_out, dtype=torch.int32)
    w0 = torch.zeros(t_out, dtype=torch.float32)
    w1 = torch.zeros(t_out, dtype=torch.float32)
    for j in range(t_out):
        row = g_up[j]
        nz = (row.abs() > 1e-8).nonzero(as_tuple=False).flatten().tolist()
        if len(nz) == 1:
            i = int(nz[0])
            idx0[j] = idx1[j] = i
            w0[j] = float(row[i].item())
            w1[j] = 0.0
        else:
            i0, i1 = int(nz[0]), int(nz[1])
            if i0 > i1:
                i0, i1 = i1, i0
            idx0[j] = i0
            idx1[j] = i1
            w0[j] = float(row[i0].item())
            w1[j] = float(row[i1].item())
    return idx0, idx1, w0, w1


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
    up_i0, up_i1, up_w0, up_w1 = _interp_up_lerp_from_g_up(g_up)
    up_i0 = up_i0.view(1, 1, -1)
    up_i1 = up_i1.view(1, 1, -1)
    up_w0 = up_w0.view(1, 1, -1)
    up_w1 = up_w1.view(1, 1, -1)
    # Downsample rows are also 2-tap (mode='linear', align_corners=False), so the same gather-lerp trick
    # avoids a wide bf16 matmul on the rad → rad_down step.
    down_i0, down_i1, down_w0, down_w1 = _interp_up_lerp_from_g_up(g_down)
    down_i0 = down_i0.view(1, 1, -1)
    down_i1 = down_i1.view(1, 1, -1)
    down_w0 = down_w0.view(1, 1, -1)
    down_w1 = down_w1.view(1, 1, -1)

    w = torch_m.l_linear.weight.data.T.contiguous().to(dtype=torch.float32)
    b = torch_m.l_linear.bias.data.to(dtype=torch.float32)
    harmonics = torch.arange(1, sg.harmonic_num + 2, dtype=torch.float32).view(1, 1, -1)
    harmonic_rand_mask = torch.ones(1, 1, harmonics.shape[-1], dtype=torch.float32)
    harmonic_rand_mask[0, 0, 0] = 0.0

    dram = ttnn.DRAM_MEMORY_CONFIG
    # ``cumsum`` along downsampled time matches ``x @ triu(1)`` on the last dim; avoids TILE cumsum drift vs PyTorch.
    cumsum_mat = torch.triu(torch.ones(t_down, t_down, dtype=torch.float32))

    return {
        "time_len": int(time_len),
        "t_down": int(t_down),
        "upsample_scale": upsample_scale,
        "sampling_rate": float(sg.sampling_rate),
        "harmonic_num": int(sg.harmonic_num),
        "sine_amp": float(torch_m.sine_amp),
        "noise_std": float(sg.noise_std),
        "voiced_threshold": float(sg.voiced_threshold),
        "flag_for_pulse": bool(sg.flag_for_pulse),
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
        "interp_up_idx0": ttnn.from_torch(
            up_i0.contiguous(), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "interp_up_idx1": ttnn.from_torch(
            up_i1.contiguous(), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "interp_up_w0": ttnn.from_torch(
            up_w0.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "interp_up_w1": ttnn.from_torch(
            up_w1.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "interp_down_idx0": ttnn.from_torch(
            down_i0.contiguous(), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "interp_down_idx1": ttnn.from_torch(
            down_i1.contiguous(), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "interp_down_w0": ttnn.from_torch(
            down_w0.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "interp_down_w1": ttnn.from_torch(
            down_w1.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        ),
        "mat_cumsum": ttnn.from_torch(
            cumsum_mat.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
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
        # Fused ``(2 * pi) * upsample_scale`` used by KokoroTtnnSineGen (one multiply on device).
        "two_pi_times_upsample": ttnn.from_torch(
            torch.tensor([[[2.0 * math.pi * upsample_scale]]], dtype=torch.float32),
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
    }
