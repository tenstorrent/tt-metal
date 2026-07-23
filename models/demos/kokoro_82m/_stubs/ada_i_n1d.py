# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_i_n1d` (hexgrad/Kokoro-82M
`predictor.F0.0.norm1`, a StyleTTS2 `AdaIN1d`).

Reference torch forward:

    class AdaIN1d(nn.Module):
        def __init__(self, style_dim, num_features):
            self.norm = nn.InstanceNorm1d(num_features, affine=True)
            self.fc = nn.Linear(style_dim, num_features * 2)
        def forward(self, x, s):
            h = self.fc(s)                    # (B, 2C)
            h = h.view(h.size(0), h.size(1), 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)   # each (B, C, 1)
            return (1 + gamma) * self.norm(x) + beta

`x` is a channels-first activation `[B, C, T]`; `s` is a `[B, style_dim]`
style vector. `self.norm` is an affine `InstanceNorm1d` (normalize each
channel over the time axis per instance, then per-channel scale/shift).

Native ttnn: the style projection is a `ttnn.linear`; instance-norm is a
reduce-over-T + normalize (same recipe as the graduated `instance_norm1d`
port); the final modulation is elementwise with broadcasting over T.
Computed in float32 for a clean PCC.
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs._lstm_scan import masked_moments

HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Bind fc / norm params and return a native ttnn forward closure."""
    m = torch_module
    norm = m.norm
    eps = float(getattr(norm, "eps", 1e-5))

    # fc: Linear(style_dim, 2*C). ttnn.linear does a @ b, so store W^T.
    fc_w_t = ttnn.as_tensor(
        m.fc.weight.detach().t().contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    fc_b = ttnn.as_tensor(
        m.fc.bias.detach().reshape(1, -1).float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if getattr(norm, "weight", None) is not None:
        num_features = int(norm.weight.shape[0])
    else:
        num_features = int(m.fc.out_features) // 2

    norm_w = norm_b = None
    if getattr(norm, "weight", None) is not None:
        norm_w = ttnn.as_tensor(
            norm.weight.detach().reshape(1, -1, 1).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        norm_b = ttnn.as_tensor(
            norm.bias.detach().reshape(1, -1, 1).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _to_ttnn(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.as_tensor(
            t,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(x, s=None, *args, **kwargs):
        if s is None:
            raise RuntimeError("ada_i_n1d forward requires style vector `s`")
        x = _to_ttnn(x)  # [B, C, T]
        s = _to_ttnn(s)  # [B, style_dim]

        # Style projection: h = s @ fc_w_t + fc_b  -> [B, 2C]
        h = ttnn.linear(s, fc_w_t, bias=fc_b)

        # gamma, beta = chunk(h.view(B, 2C, 1), 2, dim=1) -> each [B, C, 1]
        b = h.shape[0]
        gamma = ttnn.slice(h, [0, 0], [b, num_features])
        beta = ttnn.slice(h, [0, num_features], [b, 2 * num_features])
        gamma = ttnn.reshape(gamma, [b, num_features, 1])
        beta = ttnn.reshape(beta, [b, num_features, 1])

        # InstanceNorm1d over the time axis per (B, C).
        # NOTE: the reduce axis T is not tile-aligned (e.g. 25 -> padded to 32
        # in TILE_LAYOUT). Computing `mean` then `x - mean` writes `-mean` into
        # the 7 padding slots, and reducing `xc*xc` folds that padding into the
        # variance -> a *systematic* PCC ceiling (~0.988). Instead compute
        # Var = E[x^2] - E[x]^2 from SUMS over the true length: x's padding is
        # 0, so `sum(x)` and `sum(x*x)` are exact regardless of tile padding,
        # and we divide by the real T (not the padded 32).
        mv = masked_moments(device, x)  # frame-axis masked mean/var under a fixed-capacity trace
        if mv is not None:
            mean, var = mv
        else:
            n = int(x.shape[-1])
            inv_n = 1.0 / float(n)
            mean = ttnn.multiply(ttnn.sum(x, dim=2, keepdim=True), inv_n)  # [B, C, 1]
            mean_x2 = ttnn.multiply(ttnn.sum(ttnn.multiply(x, x), dim=2, keepdim=True), inv_n)
            var = ttnn.subtract(mean_x2, ttnn.multiply(mean, mean))
        xc = ttnn.subtract(x, mean)
        norm_x = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
        if norm_w is not None:
            norm_x = ttnn.add(ttnn.multiply(norm_x, norm_w), norm_b)

        # (1 + gamma) * norm_x + beta   (broadcast over T)
        out = ttnn.add(ttnn.multiply(ttnn.add(gamma, 1.0), norm_x), beta)
        return out

    return forward


def ada_i_n1d(*args, **kwargs):
    raise RuntimeError(
        "ada_i_n1d requires build(device, torch_module) to bind fc/norm params; "
        "the bare callable has no configuration."
    )
