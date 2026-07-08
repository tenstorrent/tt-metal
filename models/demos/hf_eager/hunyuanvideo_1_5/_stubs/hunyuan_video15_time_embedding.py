# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_time_embedding` of tencent/HunyuanVideo-1.5.

Reference submodule: `time_embed`, a `HunyuanVideo15TimeEmbedding` (use_meanflow
False for this checkpoint):

    timesteps_proj = self.time_proj(timestep)                      # Timesteps(256) sinusoidal
    timesteps_emb  = self.timestep_embedder(timesteps_proj)        # MLP 256 -> D -> D (SiLU)
    return timesteps_emb

Input at test time:
    timestep : (N,) 1-D — PRIMARY (arrives as a ttnn tensor; built as 1.0, bf16-exact)

Native ttnn strategy
--------------------
The weightless sinusoidal `time_proj` frequency vector is precomputed on host and
uploaded once; on device the embedding is `args = timestep[:, None] @ freq[None, :]`
(outer product as a matmul), then `concat([cos(args), sin(args)])` (flip_sin_to_cos
True). The `timestep_embedder` MLP is `matmul + bias -> SiLU -> matmul + bias`.
Float32 math with a HiFi4 config.
"""

from __future__ import annotations

import math

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"

_MAX_PERIOD = 10000


def build(device, torch_module):
    """Bind the timestep MLP + sinusoidal frequency constant; native ttnn forward."""
    import torch

    m = torch_module
    tp = m.time_proj  # Timesteps
    te = m.timestep_embedder  # TimestepEmbedding: linear_1(256->D), act(silu), linear_2(D->D)

    num_channels = int(getattr(tp, "num_channels", 256))
    flip = bool(getattr(tp, "flip_sin_to_cos", True))
    shift = float(getattr(tp, "downscale_freq_shift", 0.0))
    scale = float(getattr(tp, "scale", 1.0))
    half_dim = num_channels // 2

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def lin_weights(linear):
        w = f32(linear.weight.detach().t())
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
        return w, b

    w1, b1 = lin_weights(te.linear_1)
    w2, b2 = lin_weights(te.linear_2)

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

    def forward(timestep, timestep_r=None, *args, **kwargs):
        ts = _to_f32_device(timestep)
        n = 1
        for d in ts.shape:
            n *= int(d)
        ts = ttnn.reshape(ts, (n, 1))

        a = ttnn.matmul(ts, freq_row, compute_kernel_config=compute_config)  # (N, half)
        cos = ttnn.cos(a)
        sin = ttnn.sin(a)
        proj = ttnn.concat([cos, sin] if flip else [sin, cos], dim=-1)  # (N, 256)

        h = ttnn.matmul(proj, w1, compute_kernel_config=compute_config)
        if b1 is not None:
            h = ttnn.add(h, b1)
        h = ttnn.silu(h)
        h = ttnn.matmul(h, w2, compute_kernel_config=compute_config)
        if b2 is not None:
            h = ttnn.add(h, b2)
        return h

    return forward


def hunyuan_video15_time_embedding(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_time_embedding requires build(device, torch_module) to bind the "
        "MLP weights; the bare callable has no parameters."
    )
