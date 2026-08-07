# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Timestep and text conditioning. Frozen under LoRA, so the whole path runs with no grad."""

from __future__ import annotations

import numpy as np
import ttnn

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

_MOD_CHUNKS = 6


def timestep_features(
    timesteps,
    freq_dim: int,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    max_period: float = 10000.0,
) -> np.ndarray:
    """Sinusoidal timestep features, (B, freq_dim). Depends only on t, so host maths."""
    half = freq_dim // 2
    exponent = -np.log(max_period) * np.arange(half, dtype=np.float32) / (half - downscale_freq_shift)
    ang = np.asarray(timesteps, dtype=np.float32).reshape(-1, 1) * np.exp(exponent)[None, :]
    out = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)
    if flip_sin_to_cos:
        out = np.concatenate([out[:, half:], out[:, :half]], axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _gelu_tanh_nograd(x):
    """Wan's activation. ttml's gelu is the exact form, so route to the tanh kernel."""
    value = ttnn.gelu(x.get_value(), fast_and_approximate_mode=True)
    return ttml.autograd.create_tensor(value, False)


class _ProjectionMLP(AbstractModuleBase):
    """linear_1 -> activation -> linear_2, named to match the diffusers checkpoint."""

    def __init__(self, in_features: int, out_features: int, activation) -> None:
        super().__init__()
        self.linear_1 = LinearLayer(in_features, out_features, True, weight_init=ttml.init.normal(0.0, 0.02))
        self.linear_2 = LinearLayer(out_features, out_features, True, weight_init=ttml.init.normal(0.0, 0.02))
        self._activation = activation

    def forward(self, x):
        return self.linear_2(self._activation(self.linear_1(x)))


class WanConditioning(AbstractModuleBase):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.dim
        self.freq_dim = config.freq_dim

        self.time_embedder = _ProjectionMLP(config.freq_dim, config.dim, ttml.ops.unary.silu)
        self.time_proj = LinearLayer(
            config.dim, _MOD_CHUNKS * config.dim, True, weight_init=ttml.init.normal(0.0, 0.02)
        )
        self.text_embedder = _ProjectionMLP(config.text_dim, config.dim, _gelu_tanh_nograd)

    def forward(self, timesteps, text_embed):
        """Return (timestep_proj (B,1,6,dim), temb (B,1,1,dim), prompt (B,1,L,dim)).

        timesteps is a host sequence of length B; text_embed is the cached UMT5 output.
        """
        ctx = ttml.autograd.AutoContext.get_instance()
        previous_mode = ttml.autograd.GradMode.ENABLED
        ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        try:
            features = timestep_features(timesteps, self.freq_dim)
            batch = features.shape[0]
            features = features.reshape(batch, 1, 1, self.freq_dim)
            t_in = ttml.autograd.Tensor.from_numpy(features, ttnn.Layout.TILE, ttnn.bfloat16)

            temb = self.time_embedder(t_in)
            projected = self.time_proj(ttml.ops.unary.silu(temb))
            projected = ttml.autograd.create_tensor(
                ttnn.reshape(projected.get_value(), (batch, 1, _MOD_CHUNKS, self.dim)), False
            )
            prompt = self.text_embedder(text_embed)
        finally:
            ctx.set_gradient_mode(previous_mode)

        return projected, temb, prompt
