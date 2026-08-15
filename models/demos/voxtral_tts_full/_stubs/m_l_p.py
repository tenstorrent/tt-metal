# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TT-NN port of `VoxtralMLP` (Voxtral-TTS backbone, Block 1).

Reference: `modeling_layers.VoxtralMLP.forward` -> `voxtral_common_ref.swiglu`:

    swiglu(h, w1, w2, w3) = w2(silu(w1 h) * w3 h)

The module names those three `gate_proj` (w1), `down_proj` (w2) and `up_proj` (w3), so the
gated branch is gate/w1 and the passthrough is up/w3 -- a naming that is easy to swap by
accident, and swapping it still produces plausible-looking numbers.

The projections are bare `nn.Parameter`s, not `nn.Linear`s. `F.linear(x, W)` is `x @ W.T`, so
the transpose is folded into the staged weights instead of being paid per call. All three are
bias-free; the inner width is 9216 against a 3072 residual stream.
"""
from __future__ import annotations

import torch
import ttnn

from models.demos.voxtral_tts_full._stubs.attention import linear, stage_weight


class TtVoxtralMLP:
    def __init__(self, gate, down, up):
        self.gate, self.down, self.up = gate, down, up

    @classmethod
    def build(cls, device, torch_module, dtype=ttnn.bfloat16):
        # `stage_weight` folds in the F.linear transpose's counterpart and, at ttnn.float32,
        # stages the hi/lo bfloat16 pair the model's `linear` multiplies exactly -- same four
        # bytes per parameter, 3.7x tighter product. See `_stubs/attention.py`.
        stage = lambda t: stage_weight(device, t.detach().float().t(), dtype)  # noqa: E731
        return cls(stage(torch_module.gate_proj), stage(torch_module.down_proj), stage(torch_module.up_proj))

    def __call__(self, h, *args, **kwargs):
        gated = ttnn.mul(ttnn.silu(linear(h, self.gate)), linear(h, self.up))
        return linear(gated, self.down)


def build(device, torch_module=None, **kwargs):
    return TtVoxtralMLP.build(device, torch_module, **kwargs)


def m_l_p(device, torch_module=None, **kwargs):
    return TtVoxtralMLP.build(device, torch_module, **kwargs)
