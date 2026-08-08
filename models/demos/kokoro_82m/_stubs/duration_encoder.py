# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `duration_encoder` (hexgrad/Kokoro-82M
`predictor.text_encoder`, a StyleTTS2 `DurationEncoder`).

Reference torch forward interleaves bidirectional LSTMs with AdaLayerNorm blocks,
threading the style vector back in after each stage (masking is a no-op for a
single, fully-valid sequence — the captured mask is all-False):

    x = cat([x^T, style_broadcast], -1)                 # [B, T, d_model+sty]
    for (lstm, adln) in pairs:
        x = lstm(x)                                      # [B, T, d_model]
        x = adln(x, style)                               # AdaLayerNorm
        x = cat([x, style_broadcast], -1)               # [B, T, d_model+sty]
    return x                                             # [B, T, d_model+sty]

Native ttnn: the bidirectional LSTM is unrolled cell-by-cell (gate = xᵀWih +
hᵀWhh + b; i/f/o = sigmoid, g = tanh; c = f·c + i·g; h = o·tanh(c)), forward and
reverse directions concatenated. AdaLayerNorm reuses the graduated native
`ada_layer_norm` port. All fp32 (HiFi4) for a clean PCC.
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs._lstm_scan import run_bilstm
from models.demos.kokoro_82m._stubs.ada_layer_norm import build as _build_adln

_DRAM = ttnn.DRAM_MEMORY_CONFIG


HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Bind LSTM/AdaLayerNorm params and return a native ttnn forward closure."""

    m = torch_module

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _t(x):
        return ttnn.from_torch(
            x.contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )

    def _build_lstm(lstm):
        H = int(lstm.hidden_size)

        def _dir(suffix):
            wih = getattr(lstm, f"weight_ih_l0{suffix}").detach().float()  # [4H, in]
            whh = getattr(lstm, f"weight_hh_l0{suffix}").detach().float()  # [4H, H]
            bih = getattr(lstm, f"bias_ih_l0{suffix}").detach().float()  # [4H]
            bhh = getattr(lstm, f"bias_hh_l0{suffix}").detach().float()
            return {
                "wih_t": _t(wih.t()),  # [in, 4H]
                "whh_t": _t(whh.t()),  # [H, 4H]
                "bias": _t((bih + bhh).reshape(1, 1, -1)),  # [1,1,4H]
                "H": H,
            }

        fwd = _dir("")
        rev = _dir("_reverse")
        return {"fwd": fwd, "rev": rev, "H": H}

    lstms = []
    adlns = []
    for block in m.lstms:
        if type(block).__name__ == "AdaLayerNorm":
            adlns.append(_build_adln(device, block))
        else:
            lstms.append(_build_lstm(block))

    def _lstm(x, d, T):
        # x: [1, T, in]. Bidirectional -> [1, T, 2H]. Uses the shared masked-capable scan.
        return run_bilstm(device, compute_config, x, d["fwd"], d["rev"], d["H"])

    def _to_ttnn(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(
            t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
        )

    def forward(x, style=None, text_lengths=None, m=None, *args, **kwargs):
        if style is None:
            raise RuntimeError("duration_encoder forward requires `style`")
        x = _to_ttnn(x)  # [B, C, T]
        style = _to_ttnn(style)  # [B, sty]

        xbtc = ttnn.transpose(x, 1, 2)  # [B, T, C]
        B, T, C = int(xbtc.shape[0]), int(xbtc.shape[1]), int(xbtc.shape[2])
        sty = int(style.shape[-1])

        # style broadcast to [B, T, sty]
        s_bc = ttnn.repeat(ttnn.reshape(style, (B, 1, sty)), ttnn.Shape((1, T, 1)))

        xt = ttnn.concat([xbtc, s_bc], dim=-1)  # [B, T, C+sty]

        for i in range(len(lstms)):
            h = _lstm(xt, lstms[i], T)  # [B, T, d_model]
            a = adlns[i](h, style)  # AdaLayerNorm -> [B, T, d_model]
            xt = ttnn.concat([a, s_bc], dim=-1)  # [B, T, d_model+sty]

        return xt

    return forward


def duration_encoder(*args, **kwargs):
    raise RuntimeError(
        "duration_encoder requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
