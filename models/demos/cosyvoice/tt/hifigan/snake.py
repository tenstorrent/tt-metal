# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Snake activation: x + sin^2(alpha*x)/alpha, with alpha learned per channel.

    Snake(x) = x + (1/alpha) * sin^2(alpha * x)

TTNN has no native `snake`, so it is composed from primitives here. The composed
form costs five dispatches and four intermediates per activation:

    t = multiply(x, alpha)      ->  sin(t)  ->  square(.)
      ->  multiply(., 1/alpha)  ->  add(x, .)

HiFT applies this a lot. Each ResBlock runs 6 Snakes (convs1 and convs2 over 3
dilations); there are 2 stages x 3 kernels = 6 main ResBlocks plus 2 source
ResBlocks = 8 ResBlocks, so ~48 Snake activations per vocoder call, i.e. ~240
dispatches and ~192 intermediates at 512 channels and audio-rate length.

That is the case for a native `ttnn.snake` (03_plan.md P1) -- and the case is
broader than CosyVoice: Snake is the standard activation in BigVGAN and its
derivatives, so any HiFi-GAN-family vocoder brought up on Tenstorrent pays this.
This module is written so a native op can replace `__call__` without touching
anything else.

Two implementation notes that matter numerically:

1. `alpha_logscale=False` for CosyVoice, so alpha is used directly rather than
   exponentiated. The reference's Snake supports both; only the direct form is
   reachable from this checkpoint's config.
2. `1/alpha` is folded ON HOST at construction. The reference computes
   `reciprocal(alpha + 1e-9)` every forward pass; alpha is frozen at inference, so
   that is a per-call division of a [1, C, 1] tensor for a constant result. The
   epsilon is kept -- dropping it would change the result if any alpha is 0.
"""
from __future__ import annotations

import torch

import ttnn


class TtSnake:
    """Per-channel Snake.

    `alpha` is [C]. Activations are **channels-last `[B, T, C]`**, matching
    conv.py's convention for the whole vocoder, so alpha broadcasts over the LAST
    axis. The reference works in `[B, C, T]` and broadcasts over the middle axis;
    getting that backwards does not produce a wrong answer, it produces a
    TT_FATAL broadcasting-rule violation -- which is the good outcome, and is how
    this was caught.
    """

    EPS = 1e-9  # matches the reference's alpha.reciprocal() guard

    def __init__(self, device, alpha: torch.Tensor, dtype=ttnn.bfloat16, alpha_logscale: bool = False):
        if alpha_logscale:
            alpha = torch.exp(alpha)
        alpha = alpha.detach().float().reshape(1, 1, -1)

        self.device = device
        self.channels = alpha.shape[-1]
        self.alpha = ttnn.from_torch(alpha, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        # Folded on host: constant at inference, so the reference's per-call
        # reciprocal is pure overhead.
        self.inv_alpha = ttnn.from_torch(1.0 / (alpha + self.EPS), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, x):
        """x: ttnn [B, T, C] (channels-last) -> ttnn [B, T, C].

        Intermediates are deallocated as they die; at 512 channels and audio-rate
        length these are not small, and L1 pressure is the constraint that decides
        whether the vocoder fits (CLAUDE.md Stage 5 sec.4).
        """
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
        """cosyvoice.transformer.activation.Snake.forward (:73), verbatim in shape
        semantics. HiFT constructs it at generator.py:102,106 with
        alpha_logscale=False, so only the direct-alpha branch is reachable here.

        Defaults to the reference's [B, C, T]; pass channels_last=True to compare
        against the device path directly.
        """
        a = torch.exp(alpha) if alpha_logscale else alpha
        a = a.reshape(1, 1, -1) if channels_last else a.reshape(1, -1, 1)
        return x + (1.0 / (a + TtSnake.EPS)) * torch.sin(a * x).pow(2)
