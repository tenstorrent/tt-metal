# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `instance_norm1d` (coqui/XTTS-v2
`hifigan_decoder.speaker_encoder.instancenorm`).

The submodule is a plain `torch.nn.InstanceNorm1d` (affine=False,
track_running_stats=False, eps=1e-5) applied to a `[B, C, L]` tensor. It
normalizes each channel of each instance independently over the time axis:

    y[b, c, t] = (x[b, c, t] - mean_bc) / sqrt(var_bc + eps)

with `mean_bc`, `var_bc` the (biased / population) mean and variance over `t`.
When `affine` is set it additionally scales/shifts per channel with the learned
`weight`/`bias`.

Native ttnn: reduce over the L axis with `ttnn.mean`, then normalize with
`subtract` / `multiply` / `rsqrt`. Computed in float32 for a clean PCC.
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs._lstm_scan import masked_moments

HF_MODEL_ID = "coqui/XTTS-v2"


def build(device, torch_module):
    """Bind eps / affine params and return a native ttnn forward closure."""
    m = torch_module
    eps = float(getattr(m, "eps", 1e-5))
    affine = bool(getattr(m, "affine", False))

    weight = bias = None
    if affine and getattr(m, "weight", None) is not None:
        weight = ttnn.as_tensor(
            m.weight.detach().reshape(1, -1, 1).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bias = ttnn.as_tensor(
            m.bias.detach().reshape(1, -1, 1).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.as_tensor(
                x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        # x: [B, C, L]; normalize over the last (time) axis per (B, C).
        mv = masked_moments(device, x)  # frame-axis masked mean/var under a fixed-capacity trace
        if mv is not None:
            mean, var = mv
            xc = ttnn.subtract(x, mean)
        else:
            mean = ttnn.mean(x, dim=2, keepdim=True)  # [B, C, 1]
            xc = ttnn.subtract(x, mean)
            var = ttnn.mean(ttnn.multiply(xc, xc), dim=2, keepdim=True)
        y = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
        if weight is not None:
            y = ttnn.add(ttnn.multiply(y, weight), bias)
        return y

    return forward


def instance_norm1d(*args, **kwargs):
    raise RuntimeError(
        "instance_norm1d requires build(device, torch_module) to bind eps/affine "
        "params; the bare callable has no configuration."
    )
