# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 VAE Snake1d activation (TTTv2-pattern custom module).

Reference: Snake1d in diffusers AutoencoderOobleck (autoencoder_oobleck.py), logscale=True:

    a = exp(alpha);  b = exp(beta)                    (per-channel, [1, C, 1])
    y = x + 1/(b + 1e-9) * sin(a * x)^2

The `logscale=True` default means the stored alpha/beta are LOG-domain: the effective scale is
`exp(alpha)` / `exp(beta)`. exp(beta) is always positive (~1) so 1/(exp(beta)+1e-9) is well
conditioned (unlike the raw-beta form, which explodes when beta is a small negative number).
We fold the `exp` into the host weight-prep so the device path stays 3 elementwise ops.

alpha, beta are learned parameters of shape [1, C, 1]; the activation is applied to a
[B, C, T] (channels-second) audio feature map. This is a pure elementwise op with a
channel-broadcast scale, so there's no TTTv2 library match — we implement it directly with
ttnn elementwise ops following the LightweightModule + Config contract.

We operate on the transposed [B, T, C] layout used throughout the TT VAE decoder (so C is the
last, tile-friendly dim and alpha/beta broadcast as [1, 1, C]); the decoder conv wrappers keep
data in [B, T, C] between ops and only transpose to [B, C, T] conceptually for conv math.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight


@dataclass
class Snake1dConfig:
    # alpha, beta reshaped to [1, 1, C] (broadcast over B, T) for the [B, T, C] layout.
    alpha: LazyWeight
    beta: LazyWeight
    channels: int

    mesh_device: ttnn.MeshDevice | None = None

    def resolved(self) -> "Snake1dConfig":
        if self.mesh_device is None:
            self.mesh_device = self.alpha.device
        return self


def snake_alpha_beta_to_lazy(param: torch.Tensor, device, dtype=ttnn.bfloat16, *, logscale: bool = True) -> LazyWeight:
    """Reshape a Snake1d [1, C, 1] parameter to [1, 1, C] LazyWeight for [B, T, C] broadcasting.

    With ``logscale=True`` (the diffusers default) the effective scale is exp(param); we fold that
    exp on host so the device forward is a plain multiply/reciprocal.
    """
    p = param.detach().float()
    if logscale:
        p = torch.exp(p)
    c = p.numel()
    src = p.reshape(1, 1, c).contiguous()
    return LazyWeight(
        source=src, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


class Snake1d(LightweightModule):
    """Snake activation over [B, T, C]: x + (beta+1e-9)^-1 * sin(alpha*x)^2."""

    EPS = 1e-9

    def __init__(self, config: Snake1dConfig):
        super().__init__()
        self.config = config.resolved()
        self.alpha = config.alpha.get_device_weight()  # [1,1,C]
        self.beta = config.beta.get_device_weight()  # [1,1,C]

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # sin(alpha * x)^2
        ax = ttnn.multiply(x, self.alpha)
        s = ttnn.sin(ax)
        s2 = ttnn.multiply(s, s)
        # (beta + 1e-9)^-1  * sin^2
        beta_eps = ttnn.add(self.beta, self.EPS)
        inv_beta = ttnn.reciprocal(beta_eps)
        term = ttnn.multiply(s2, inv_beta)
        out = ttnn.add(x, term)

        ttnn.deallocate(ax)
        ttnn.deallocate(s)
        ttnn.deallocate(s2)
        ttnn.deallocate(beta_eps)
        ttnn.deallocate(inv_beta)
        ttnn.deallocate(term)
        return out
