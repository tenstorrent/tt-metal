# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `timesteps` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder.time_text_embed.time_proj`, a diffusers
`Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0,
scale=1)` — a *weightless* sinusoidal timestep projection. Its forward is
`diffusers.get_timestep_embedding`:

    half = num_channels // 2
    exponent = -log(max_period) * arange(half) / (half - downscale_freq_shift)
    freq     = exp(exponent) * scale                     # (half,)
    args     = timesteps[:, None] * freq[None, :]         # (N, half)
    emb      = cat([sin(args), cos(args)], dim=-1)        # (N, 2*half)
    if flip_sin_to_cos: emb = cat([cos-half, sin-half])   # -> [cos, sin]

Inputs at test time:
    timesteps : (N,) 1-D — PRIMARY (arrives as a ttnn tensor); the harness
                builds bf16-exact integer values so the bf16 cast is lossless.

Native ttnn strategy
--------------------
There are no learned weights. The frequency vector is a host constant,
precomputed once at build time and uploaded as a (1, half) row. On device the
outer product `timesteps[:, None] * freq[None, :]` is done as a matmul
(`(N,1) @ (1,half)` — robust for any N), then `concat([cos(args), sin(args)])`
(flip_sin_to_cos=True). All device math is float32 under a HiFi4 config.

This mirrors the already-graduated sibling `combined_timestep_text_proj_embeddings`
stub, whose `time_proj` sub-computation is byte-for-byte this same code path.
"""

from __future__ import annotations

import math

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"

# Defaults for Timesteps(num_channels=256, flip_sin_to_cos=True,
# downscale_freq_shift=0, scale=1); read from the module when available.
_NUM_CHANNELS = 256
_FLIP_SIN_TO_COS = True
_DOWNSCALE_FREQ_SHIFT = 0.0
_SCALE = 1.0
_MAX_PERIOD = 10000


def build(device, torch_module):
    """Precompute the sinusoidal frequency constant; return a native forward."""
    import torch

    tp = torch_module
    num_channels = int(getattr(tp, "num_channels", _NUM_CHANNELS))
    flip = bool(getattr(tp, "flip_sin_to_cos", _FLIP_SIN_TO_COS))
    shift = float(getattr(tp, "downscale_freq_shift", _DOWNSCALE_FREQ_SHIFT))
    scale = float(getattr(tp, "scale", _SCALE))
    half_dim = num_channels // 2

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Weightless frequency vector (host constant), uploaded as a (1, half) row.
    exponent = -math.log(_MAX_PERIOD) * torch.arange(0, half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - shift)
    freq = torch.exp(exponent) * scale  # (half_dim,)
    freq_row = f32(freq.reshape(1, half_dim))  # (1, half_dim)

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(x, *args, **kwargs):
        ts = _to_f32_device(x)
        n = 1
        for d in ts.shape:
            n *= int(d)
        ts = ttnn.reshape(ts, (n, 1))  # (N, 1)

        a = ttnn.matmul(ts, freq_row, compute_kernel_config=compute_config)  # (N, half)
        cos = ttnn.cos(a)
        sin = ttnn.sin(a)
        # flip_sin_to_cos=True -> [cos, sin]; else [sin, cos]
        return ttnn.concat([cos, sin] if flip else [sin, cos], dim=-1)  # (N, 256)

    return forward


def timesteps(*args, **kwargs):
    raise RuntimeError(
        "timesteps requires build(device, torch_module) to read the Timesteps "
        "config (num_channels / flip_sin_to_cos / downscale_freq_shift / scale); "
        "the bare callable has no parameters."
    )
