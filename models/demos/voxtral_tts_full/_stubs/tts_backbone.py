# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `backbone` (`VoxtralTtsBackbone`) -- Block 1, the 3.4B AR backbone.

`forward` is the causal prefill with no cache: 26 pre-norm GQA + SwiGLU layers over the input
embeddings, then the final RMSNorm.  Each layer is the same `TtBackboneLayer` the `attention`,
`m_l_p` and `decoder_layer` stubs are built from
(`models/demos/voxtral_tts_full/tt_backbone.py`), so this component composes exactly what those
three graduated on rather than a second implementation of it.

The RoPE cos/sin tables and the causal mask are built ONCE and shared by all 26 layers -- they
depend only on position, not on the layer -- which is also what keeps them out of the probed
forward (`ttnn.from_torch` is 2 torch ops and `native_probe` graduates at 0).

Weight staging: the released checkpoint is bfloat16, so `stage_weight` finds every weight exactly
representable there and stages it as bf16.  That is lossless against this golden AND halves the
device footprint (6 GB instead of 12 GB for 3.03B parameters).  Activations stay float32
regardless: a matmul with an fp32 activation is 4.9e-4 relative whichever way the weight is
staged, but 2.4e-3 once the activation itself is rounded to bf16 -- and over 26 layers that is
the difference that shows.
"""

from __future__ import annotations

from models.demos.voxtral_tts_full.tt_backbone import (
    TtBackboneLayer,
    TtBackboneTables,
    TtRMSNorm,
)


class TtVoxtralTtsBackbone:
    def __init__(self, layers, norm):
        self.layers = layers
        self.norm = norm

    @classmethod
    def build(cls, device, torch_module):
        tables = TtBackboneTables(device)
        # The RoPE row permutation is structural -- identical for every layer -- so it is
        # asserted against layer 0's real weights and taken on trust for the other 25.
        layers = [
            TtBackboneLayer.from_module(device, layer, tables, verify=(i == 0))
            for i, layer in enumerate(torch_module.layers)
        ]
        return cls(layers, TtRMSNorm.from_module(device, torch_module.norm))

    def __call__(self, inputs_embeds):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, causal=True)
        return self.norm(x)


def build(device, torch_module=None):
    return TtVoxtralTtsBackbone.build(device, torch_module)


def tts_backbone(device, torch_module=None):
    return TtVoxtralTtsBackbone.build(device, torch_module)
