# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Timestep and text conditioning. Frozen under LoRA, so the whole path runs no-grad."""

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
    """Sinusoidal timestep features, (B, freq_dim). Depends only on t, so host-side."""
    half = freq_dim // 2
    exponent = -np.log(max_period) * np.arange(half, dtype=np.float32) / (half - downscale_freq_shift)
    ang = np.asarray(timesteps, dtype=np.float32).reshape(-1, 1) * np.exp(exponent)[None, :]
    out = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)
    if flip_sin_to_cos:
        out = np.concatenate([out[:, half:], out[:, :half]], axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


# TODO: Use ttml.ops.unary.gelu once it takes a variant (bmijanovicTT)
# Issue: #53776
# ttml hardcodes the exact erf form, so tanh is only reachable through raw ttnn.
def _gelu_tanh_nograd(x):
    """ttml's gelu is the exact erf form, so route to the tanh kernel."""
    value = ttnn.gelu(x.get_value(), variant=ttnn.GeluVariant.Tanh)
    return ttml.autograd.create_tensor(value, False)


class _ProjectionMLP(AbstractModuleBase):
    """linear_1 -> activation -> linear_2, named to match the diffusers checkpoint."""

    def __init__(self, in_features: int, out_features: int, activation, weight_init) -> None:
        super().__init__()
        self.linear_1 = LinearLayer(in_features, out_features, True, weight_init=weight_init)
        self.linear_2 = LinearLayer(out_features, out_features, True, weight_init=weight_init)
        self._activation = activation

    def forward(self, x):
        return self.linear_2(self._activation(self.linear_1(x)))


class WanConditioning(AbstractModuleBase):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.dim
        self.freq_dim = config.freq_dim

        init = config.weight_init()
        self.time_embedder = _ProjectionMLP(config.freq_dim, config.dim, ttml.ops.unary.silu, init)
        self.time_proj = LinearLayer(config.dim, _MOD_CHUNKS * config.dim, True, weight_init=init)
        self.text_embedder = _ProjectionMLP(config.text_dim, config.dim, _gelu_tanh_nograd, init)

    def forward(self, timesteps, text_embed):
        """(timestep_proj (B,1,6,dim), temb (B,1,1,dim), prompt (B,1,L,dim))."""
        ctx = ttml.autograd.AutoContext.get_instance()
        previous_mode = ctx.get_gradient_mode()
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
