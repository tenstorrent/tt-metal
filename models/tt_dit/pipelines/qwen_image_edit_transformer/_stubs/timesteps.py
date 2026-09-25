# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of the sinusoidal timestep projection (`time_proj`, diffusers Timesteps /
get_timestep_embedding; QwenImage: num_channels=256, flip_sin_to_cos=True, shift=0, scale=1000):

    freq = exp(-ln(max_period) * arange(half) / (half - shift))
    emb  = scale * (t[:, None] * freq[None, :])
    out  = [cos(emb) | sin(emb)]   (flip_sin_to_cos; [sin | cos] otherwise), zero-padded if odd

`freq` is a constant computed once at build time with the reference's float32 expression; the
per-call products and sin/cos run on device in float32.
"""

from __future__ import annotations

import math

import torch

import ttnn


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


class TtTimesteps:
    def __init__(self, device, torch_module, max_period=10000):
        self.device = device
        self.dim = int(torch_module.num_channels)
        self.flip = bool(torch_module.flip_sin_to_cos)
        self.scale = float(torch_module.scale)
        shift = float(torch_module.downscale_freq_shift)
        half = self.dim // 2
        exponent = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
        freq = torch.exp(exponent / (half - shift))
        self.freq = _replicated(freq.reshape(1, half), device)

    def __call__(self, timesteps, **_unused):
        t = timesteps
        if not isinstance(t, ttnn.Tensor):
            t = _replicated(t.to(torch.float32), self.device)
        if t.dtype != ttnn.float32:
            t = ttnn.typecast(t, ttnn.float32)
        N = t.shape[0]
        t = ttnn.reshape(t, (N, 1))
        emb = ttnn.multiply(ttnn.multiply(t, self.freq), self.scale)
        s, c = ttnn.sin(emb), ttnn.cos(emb)
        out = ttnn.concat([c, s] if self.flip else [s, c], dim=-1)
        if self.dim % 2 == 1:
            out = ttnn.pad(out, [(0, 0), (0, 1)], value=0.0)
        return out


def build(device, torch_module=None):
    return TtTimesteps(device, torch_module)


def timesteps(device, torch_module=None):
    return TtTimesteps(device, torch_module)
