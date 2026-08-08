# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_layer_norm` (hexgrad/Kokoro-82M
`predictor.text_encoder.lstms.1`, a StyleTTS2 `AdaLayerNorm`).

Reference torch forward:

    class AdaLayerNorm(nn.Module):
        def __init__(self, style_dim, channels, eps=1e-5):
            self.channels = channels
            self.eps = eps
            self.fc = nn.Linear(style_dim, channels * 2)
        def forward(self, x, s):
            x = x.transpose(-1, -2); x = x.transpose(1, -1)   # net identity for 3D x
            h = self.fc(s)
            h = h.view(h.size(0), h.size(1), 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)     # (B, C, 1)
            gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)  # (B, 1, C)
            x = F.layer_norm(x, (self.channels,), eps=self.eps)          # over last dim
            x = (1 + gamma) * x + beta
            return x.transpose(1, -1).transpose(-1, -2)         # net identity

For a 3D `[B, T, C]` activation the transpose pairs cancel, so this is a plain
LayerNorm over the channel (last) axis with NO built-in affine, followed by a
style-conditioned modulation `(1 + gamma) * x + beta` where `gamma`, `beta` are
`[B, 1, C]` slices of `fc(s)`. Computed in float32 for a clean PCC.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Bind fc params / eps and return a native ttnn forward closure."""
    m = torch_module
    eps = float(getattr(m, "eps", 1e-5))
    channels = int(getattr(m, "channels", m.fc.out_features // 2))

    fc_w_t = ttnn.as_tensor(
        m.fc.weight.detach().t().contiguous().float(),  # [style_dim, 2C]
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    fc_b = ttnn.as_tensor(
        m.fc.bias.detach().reshape(1, -1).float(),  # [1, 2C]
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
            raise RuntimeError("ada_layer_norm forward requires style vector `s`")
        x = _to_ttnn(x)  # [B, T, C]
        s = _to_ttnn(s)  # [B, style_dim]

        h = ttnn.linear(s, fc_w_t, bias=fc_b)  # [B, 2C]
        b = h.shape[0]
        gamma = ttnn.reshape(ttnn.slice(h, [0, 0], [b, channels]), [b, 1, channels])
        beta = ttnn.reshape(ttnn.slice(h, [0, channels], [b, 2 * channels]), [b, 1, channels])

        # LayerNorm over the channel (last) axis, no built-in affine.
        mean = ttnn.mean(x, dim=2, keepdim=True)  # [B, T, 1]
        xc = ttnn.subtract(x, mean)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=2, keepdim=True)
        xn = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))

        # (1 + gamma) * xn + beta   (broadcast over T)
        return ttnn.add(ttnn.multiply(ttnn.add(gamma, 1.0), xn), beta)

    return forward


def ada_layer_norm(*args, **kwargs):
    raise RuntimeError(
        "ada_layer_norm requires build(device, torch_module) to bind fc params; "
        "the bare callable has no configuration."
    )
