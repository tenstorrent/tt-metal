# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `backbone.layers.0` (`VoxtralDecoderLayer`).

    x -> input_layernorm -> self_attn -> +x -> post_attention_layernorm -> mlp -> +x

The layer math lives in `models/demos/voxtral_tts_full/tt_backbone.py`, shared with the
`attention`, `m_l_p` and `tts_backbone` stubs -- the decomposition split one layer into its
parts, so re-deriving the same arithmetic four times would only give it four places to drift.

`cis` and `bias` arrive as host torch objects and are read for PRESENCE only: one
`ttnn.from_torch` inside the forward is 2 torch ops, and `native_probe` graduates at 0.  Both are
pure functions of the sequence length, so `TtBackboneTables` precomputes them in `build()` (not
probed) and slices them on device.
"""

from __future__ import annotations

from models.demos.voxtral_tts_full.tt_backbone import TtBackboneLayer, TtBackboneTables


class TtVoxtralDecoderLayer:
    def __init__(self, layer):
        self.layer = layer

    @classmethod
    def build(cls, device, torch_module):
        tables = TtBackboneTables(device)
        return cls(TtBackboneLayer.from_module(device, torch_module, tables))

    def __call__(self, x, cis=None, bias=None, cache=None):
        return self.layer(x, causal=bias is not None)


def build(device, torch_module=None):
    return TtVoxtralDecoderLayer.build(device, torch_module)


def decoder_layer(device, torch_module=None):
    return TtVoxtralDecoderLayer.build(device, torch_module)
