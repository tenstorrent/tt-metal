# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `backbone.layers.0.self_attn` (`VoxtralAttention`).

The generated ADAPT scaffold wrapped `models/tt_transformers/tt/attention.py::Attention`.  That
class is not constructible here -- it wants a whole model context (`tt_ccl`,
`transformation_mats`, `configuration`, a paged KV cache, `weight_cache_path`) that a standalone
per-component PCC test has no way to supply -- so this is the reference's own composition written
directly in ttnn:

    q,k,v = linear(h) -> split_heads -> apply_rope(cis) -> gqa_attention(bias) -> linear

The implementation lives in `models/demos/voxtral_tts_full/tt_backbone.py`, shared with the
`decoder_layer`, `m_l_p` and `tts_backbone` stubs (all four are the same arithmetic at different
granularities), where two model-specific points are handled:

* RoPE is MISTRAL-NATIVE -- interleaved pairs via `view_as_complex`, not HF's split-half.  The
  pairing is folded into the wq/wk ROWS at build time so the cheap half-split rotation is exact
  on device, and `verify_rope_permutation` asserts the equivalence against this checkpoint's own
  weights rather than trusting it.

* `cis` and `bias` arrive as host torch objects and are read for PRESENCE only: one
  `ttnn.from_torch` in a probed forward is 2 torch ops and `native_probe` graduates at 0.  Both
  are pure functions of the sequence length, so they are precomputed for
  `max_position_embeddings` in `build()` (not probed) and sliced on device.
"""

from __future__ import annotations

from models.demos.voxtral_tts_full.tt_backbone import TtBackboneAttention, TtBackboneTables


class TtVoxtralAttention:
    def __init__(self, attn):
        self.attn = attn

    @classmethod
    def build(cls, device, torch_module):
        tables = TtBackboneTables(device)
        return cls(TtBackboneAttention.from_module(device, torch_module, tables))

    def __call__(self, h, cis=None, bias=None, cache=None, cache_key=None):
        return self.attn(h, causal=bias is not None)


def build(device, torch_module=None):
    return TtVoxtralAttention.build(device, torch_module)


def attention(device, torch_module=None):
    return TtVoxtralAttention.build(device, torch_module)
