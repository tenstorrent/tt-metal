# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `prosody_predictor` (hexgrad/Kokoro-82M `predictor`, a
StyleTTS2 `ProsodyPredictor`) — the duration branch of its forward.

Reference forward (masking/packing are no-ops for a single full sequence,
dropout is identity in eval):

    d = text_encoder(texts, style, text_lengths, m)   # DurationEncoder -> [B,T,640]
    x, _ = lstm(pack(d))                               # nn.LSTM 640->512
    duration = duration_proj(x)                        # LinearNorm 512->50
    en = d.transpose(-1,-2) @ alignment
    return duration.squeeze(-1), en

The PCC test compares out[0] = duration (independent of `alignment`). This port
composes the graduated native children (`duration_encoder`, `l_s_t_m` LSTM cell,
`linear_norm`) and returns the duration tensor. All fp32 for a clean PCC.
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs.duration_encoder import build as _build_dur_enc
from models.demos.kokoro_82m._stubs.l_s_t_m import build as _build_lstm
from models.demos.kokoro_82m._stubs.linear_norm import build as _build_linear_norm

_DRAM = ttnn.DRAM_MEMORY_CONFIG


HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Compose the duration branch from graduated native child ports."""
    m = torch_module
    de_fwd = _build_dur_enc(device, m.text_encoder)
    lstm_fwd = _build_lstm(device, m.lstm)
    dp_fwd = _build_linear_norm(device, m.duration_proj)

    def _to_ttnn(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(
            t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
        )

    def forward(texts, style=None, text_lengths=None, alignment=None, m=None, *args, **kwargs):
        if style is None:
            raise RuntimeError("prosody_predictor forward requires `style`")
        texts = _to_ttnn(texts)  # [B, C, T]
        style = _to_ttnn(style)  # [B, sty]

        d = de_fwd(texts, style=style)  # [B, T, 640]
        x = lstm_fwd(d)  # [B, T, 512]
        duration = dp_fwd(x)  # [B, T, 50]
        return duration

    return forward


def prosody_predictor(*args, **kwargs):
    raise RuntimeError(
        "prosody_predictor requires build(device, torch_module) to compose its "
        "children; the bare callable has no parameters."
    )
