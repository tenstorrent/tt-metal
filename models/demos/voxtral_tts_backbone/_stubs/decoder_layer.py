# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `decoder_layer` (`MistralDecoderLayer`) for
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

    h  = x + self_attn(input_layernorm(x))
    out = h + mlp(post_attention_layernorm(h))

Every piece of math here is a sibling component stub — attention from
`_stubs/attention.py::TtAttention`, SwiGLU from `_stubs/m_l_p.py::TtMLP`, both
norms from `_stubs/r_m_s_norm.py::TtRMSNorm` (the canonical
`models/common/rmsnorm.py::RMSNorm`) — rather than second copies, so the
composite and the standalone components cannot drift apart.

`build` stages weights through those components (torch touches checkpoint
weights only); `__call__` is pure ttnn — `models/common/native_probe.py` counts
what actually executes, so any torch op in the forward would disqualify it.
"""
from __future__ import annotations

import ttnn

from models.demos.voxtral_tts_backbone._stubs.attention import TtAttention
from models.demos.voxtral_tts_backbone._stubs.m_l_p import TtMLP
from models.demos.voxtral_tts_backbone._stubs.r_m_s_norm import TtRMSNorm


class TtDecoderLayer:
    def __init__(self, attn, mlp, input_norm, post_attn_norm):
        self.attn = attn
        self.mlp = mlp
        self.input_norm = input_norm
        self.post_attn_norm = post_attn_norm

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("decoder_layer build needs the HF MistralDecoderLayer to read weights from")
        attn = TtAttention.build(device, torch_module.self_attn)
        mlp = TtMLP.build(device, torch_module.mlp, compute_kernel_config=attn.compute_kernel_config)
        return cls(
            attn,
            mlp,
            TtRMSNorm.build(device, torch_module.input_layernorm),
            TtRMSNorm.build(device, torch_module.post_attention_layernorm),
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        kv_cache=None,
        cache_fill=False,
        cache_pos=None,
        cache_pos_tensor=None,
        **_ignored,
    ):
        """`h = x + attn(norm1(x)); out = h + mlp(norm2(h))`, pure ttnn.

        The `kv_cache`/`cache_fill`/`cache_pos`/`cache_pos_tensor` kwargs are
        OPTIONAL and forwarded verbatim to the one graduated `TtAttention` body
        (see `_stubs/attention.py`). With them absent this forward is
        bit-for-bit what the per-component PCC test pinned.
        """
        attn_out = self.attn(
            self.input_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            cache_fill=cache_fill,
            cache_pos=cache_pos,
            cache_pos_tensor=cache_pos_tensor,
        )
        hidden_states = ttnn.add(hidden_states, attn_out)
        mlp_out = self.mlp(self.post_attn_norm(hidden_states))
        return ttnn.add(hidden_states, mlp_out)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)


# Legacy slug-named shim.
def decoder_layer(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)
