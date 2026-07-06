# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `long_cat_image_timestep_embeddings`
(diffusers ``LongCatImageTimestepEmbeddings``) for meituan-longcat/LongCat-Image.

forward(timestep, hidden_dtype)::

    proj = Timesteps(256, flip_sin_to_cos=True, shift=0)(timestep)   # sinusoidal [N,256]
    emb  = timestep_embedder(proj)                                   # Linear->SiLU->Linear [N,3072]
    return emb

Sinusoidal projection (get_timestep_embedding, flip_sin_to_cos=True, shift=0)::

    freq = exp(-log(max_period) * arange(half) / half)   # [half=128], host-constant
    args = timestep[:, None] * freq[None, :]             # [N, 128]
    proj = cat([cos(args), sin(args)], dim=-1)           # [N, 256]  (cos first: flip)

Runs fully fp32 (HiFi4 / fp32-accumulate). `freq` is precomputed on host in fp32;
the timestep drives ttnn.cos/sin. `hidden_dtype` is ignored (we stay fp32).
"""

from __future__ import annotations

import math

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
TILE = ttnn.TILE_LAYOUT


class _TimestepEmbeddings:
    def __init__(self, device, mod):
        self.device = device
        self.mod = mod.eval() if hasattr(mod, "eval") else mod
        tp = self.mod.time_proj
        self.num_channels = int(tp.num_channels)  # 256
        self.half = self.num_channels // 2  # 128
        self.flip = bool(getattr(tp, "flip_sin_to_cos", True))
        self.shift = float(getattr(tp, "downscale_freq_shift", 0.0))
        self.scale = float(getattr(tp, "scale", 1.0))
        self.max_period = int(getattr(tp, "max_period", 10000))
        self._lin = {}
        self._freq = None
        self._compute = None

    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _freqs(self):
        if self._freq is None:
            exponent = -math.log(self.max_period) * torch.arange(0, self.half, dtype=torch.float32)
            exponent = exponent / (self.half - self.shift)
            freq = torch.exp(exponent).reshape(1, self.half)  # [1, 128]
            self._freq = ttnn.from_torch(freq, dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)
        return self._freq

    def _linear(self, x, tm):
        key = id(tm)
        if key not in self._lin:
            b = tm.bias
            self._lin[key] = (
                ttnn.from_torch(
                    tm.weight.detach().to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM
                ),
                ttnn.from_torch(
                    b.detach().reshape(1, -1).to(torch.float32),
                    dtype=F32,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                )
                if b is not None
                else None,
            )
        wt, bt = self._lin[key]
        return ttnn.linear(
            x, wt, bias=bt, transpose_b=True, memory_config=DRAM, dtype=F32, compute_kernel_config=self._ck()
        )

    def __call__(self, timestep, hidden_dtype=None, **_ignored):
        t = timestep
        if isinstance(t, ttnn.Tensor):
            if t.layout != TILE:
                t = ttnn.to_layout(t, TILE)
            if t.dtype != F32:
                t = ttnn.typecast(t, F32)
        else:
            t = ttnn.from_torch(t.to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)

        # N = number of timesteps (flatten to [N, 1])
        N = 1
        for d in list(t.shape):
            N *= int(d)
        t = ttnn.reshape(t, [N, 1])

        args = ttnn.multiply(t, self._freqs())  # [N,1]*[1,128] -> [N,128]
        if self.scale != 1.0:
            args = ttnn.multiply(args, self.scale)
        cos = ttnn.cos(args)
        sin = ttnn.sin(args)
        proj = ttnn.concat([cos, sin], dim=-1) if self.flip else ttnn.concat([sin, cos], dim=-1)  # [N,256]

        te = self.mod.timestep_embedder
        x = self._linear(proj, te.linear_1)  # [N, embed_dim]
        x = ttnn.silu(x)
        x = self._linear(x, te.linear_2)  # [N, embed_dim]
        return x


def build(device, torch_module):
    """PCC-harness entry point: native TTNN LongCatImageTimestepEmbeddings."""
    return _TimestepEmbeddings(device, torch_module)
