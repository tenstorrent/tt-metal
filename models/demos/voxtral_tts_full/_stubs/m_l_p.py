# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `backbone.layers.0.mlp` (`VoxtralMLP`).

    swiglu(h, w1, w2, w3) = w2(silu(w1 h) * w3 h)

all three projections bias-free.  The module names them `gate_proj` / `down_proj` / `up_proj`
(HF's names) for the reference's `w1` / `w2` / `w3`, and they are raw `nn.Parameter`s rather than
`nn.Linear`s -- so `F.linear(h, W)` is `h @ W.T` and the transpose is folded in at build time.

The generated ADAPT scaffold wrapped `models/tt_transformers/tt/mlp.py::MLP`, which needs a whole
model context (`ModelArgs`, `tt_ccl`, a weight cache) that a standalone per-component PCC test
cannot supply; the arithmetic itself is three matmuls and lives in
`models/demos/voxtral_tts_full/tt_backbone.py`, shared with the `decoder_layer` and
`tts_backbone` stubs.
"""

from __future__ import annotations

from models.demos.voxtral_tts_full.tt_backbone import TtBackboneMLP


class TtVoxtralMLP:
    def __init__(self, mlp):
        self.mlp = mlp

    @classmethod
    def build(cls, device, torch_module):
        return cls(TtBackboneMLP.from_module(device, torch_module))

    def __call__(self, h):
        return self.mlp(h)


def build(device, torch_module=None):
    return TtVoxtralMLP.build(device, torch_module)


def m_l_p(device, torch_module=None):
    return TtVoxtralMLP.build(device, torch_module)
