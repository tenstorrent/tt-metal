# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `group_norm32` of coqui/XTTS-v2.

Reference submodule: `gpt.conditioning_encoder.attn.0.norm`, an instance of
`TTS.tts.layers.tortoise.arch_utils.GroupNorm32` (a `nn.GroupNorm` that runs its
statistics in float32). For a channels-first `[1, C, T]` input the per-group
statistics pool over (channels-in-group x T); the affine weight/bias are
per-channel. Captured shapes: in/out `[1, 1024, 259]`, groups=32.

We regroup with a row-major reshape `[1, C, T] -> [1, G, (C/G)*T]` (channels are
contiguous, so consecutive C/G channels fall in the same group row), reduce
mean/var over that axis, normalize, reshape back, and apply the per-channel
affine (broadcast over T). Config (num_groups / eps / affine) is read from the
module so this is not specific to the 32-group/1024-channel instance.
"""

from __future__ import annotations

import ttnn


def _to_tile(t):
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def _to_rm(t):
    return ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)


def build(device, torch_module):
    import torch

    m = torch_module
    num_groups = int(m.num_groups)
    eps = float(m.eps)
    weight = getattr(m, "weight", None)
    bias = getattr(m, "bias", None)

    gn_w = gn_b = None
    if weight is not None:
        c = weight.shape[0]
        gn_w = ttnn.as_tensor(
            weight.detach().reshape(1, c, 1).contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if bias is not None:
        c = bias.shape[0]
        gn_b = ttnn.as_tensor(
            bias.detach().reshape(1, c, 1).contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(x, *args, **kwargs):
        _, c, t = x.shape
        x_g = _to_tile(ttnn.reshape(_to_rm(x), (1, num_groups, (c // num_groups) * t)))

        mean = ttnn.mean(x_g, dim=2, keepdim=True)
        mean_sq = ttnn.mean(ttnn.multiply(x_g, x_g), dim=2, keepdim=True)
        var = ttnn.subtract(mean_sq, ttnn.multiply(mean, mean))
        inv_std = ttnn.rsqrt(ttnn.add(var, eps))

        x_g = ttnn.multiply(ttnn.subtract(x_g, mean), inv_std)
        x_n = _to_tile(ttnn.reshape(_to_rm(x_g), (1, c, t)))

        if gn_w is not None:
            x_n = ttnn.multiply(x_n, gn_w)
        if gn_b is not None:
            x_n = ttnn.add(x_n, gn_b)
        return x_n

    return forward


def group_norm32(x, *args, **kwargs):
    raise RuntimeError(
        "group_norm32 requires build(device, torch_module) to bind the affine "
        "weight/bias and group count; the bare callable has no parameters."
    )
