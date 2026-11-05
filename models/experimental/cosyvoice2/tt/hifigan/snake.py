# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Snake activation: x + sin^2(alpha*x)/alpha, with alpha learned per channel.

Verbatim from `cosyvoice.transformer.activation.Snake.forward` (confirmed against
upstream source, not assumed):

    x = x + (1.0 / (alpha + eps)) * pow(sin(x * alpha), 2)

with `alpha_logscale=False` for every HiFT ResBlock (CosyVoice1 and CosyVoice2
both construct `Snake(channels, alpha_logscale=False)`), so only the direct-alpha
branch is reachable from a real checkpoint.

TTNN has no native `snake`, so it is composed from primitives:

    t = multiply(x, alpha)  ->  sin(t)  ->  square(.)
      ->  multiply(., 1/alpha)  ->  add(x, .)

`1/alpha` is folded on host at construction -- alpha is frozen at inference, so
recomputing its reciprocal every forward pass (as the torch reference does) is
pure overhead here.
"""

from __future__ import annotations

import torch

import ttnn


class TtSnake:
    """Per-channel Snake.

    `alpha` is `[C]`. Activations are channels-last `[B, T, C]` (matching conv.py's
    convention for the whole vocoder), so alpha broadcasts over the last axis.
    """

    EPS = 1e-9  # matches Snake's `no_div_by_zero` guard

    def __init__(self, device, alpha: torch.Tensor, dtype=ttnn.bfloat16, alpha_logscale: bool = False):
        if alpha_logscale:
            alpha = torch.exp(alpha)
        alpha = alpha.detach().float().reshape(1, 1, -1)

        self.device = device
        self.channels = alpha.shape[-1]
        self.alpha = ttnn.from_torch(alpha, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.inv_alpha = ttnn.from_torch(1.0 / (alpha + self.EPS), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, x):
        """x: ttnn [B, T, C] -> ttnn [B, T, C]."""
        t = ttnn.multiply(x, self.alpha)
        s = ttnn.sin(t)
        ttnn.deallocate(t)
        s2 = ttnn.square(s)
        ttnn.deallocate(s)
        scaled = ttnn.multiply(s2, self.inv_alpha)
        ttnn.deallocate(s2)
        out = ttnn.add(x, scaled)
        ttnn.deallocate(scaled)
        return out

    @staticmethod
    def torch_reference(
        x: torch.Tensor, alpha: torch.Tensor, alpha_logscale: bool = False, channels_last: bool = False
    ) -> torch.Tensor:
        """Snake.forward, verbatim in shape semantics. Defaults to [B, C, T]
        (upstream's own convention); pass channels_last=True to compare against
        the device path directly."""
        a = torch.exp(alpha) if alpha_logscale else alpha
        a = a.reshape(1, 1, -1) if channels_last else a.reshape(1, -1, 1)
        return x + (1.0 / (a + TtSnake.EPS)) * torch.sin(a * x).pow(2)
