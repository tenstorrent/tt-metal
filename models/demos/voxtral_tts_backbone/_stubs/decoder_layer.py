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
            # `out_sharded`: at decode the norm and the fused QKV projection are
            # built on the SAME core grid, so the norm's result is already the
            # operand the projection wants. Handing it over in place skips the
            # sharded->interleaved on the way out and the interleaved->sharded
            # straight back in -- two launches per layer for a tensor that never
            # needed to leave L1. Prefill ignores the flag.
            self.input_norm(hidden_states, out_sharded=True),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            cache_fill=cache_fill,
            cache_pos=cache_pos,
            cache_pos_tensor=cache_pos_tensor,
        )
        # Both residual adds write STRAIGHT into the shard the norm that reads
        # them wants. The stream is one tile row of 3072 values that every op in
        # the block already touches in L1; leaving it interleaved between them
        # meant each norm re-opened the same shard from DRAM, once per norm, 26
        # layers deep. `_residual_memory_config` is None when this block is not
        # on the decode path, and then these are the plain adds they were.
        residual_cfg = self._residual_memory_config(hidden_states)
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=residual_cfg)
        # Same hand-over as the attention norm above: the MLP's gate/up plan is
        # built on this norm's core grid, so its shard is the operand they want.
        mlp_out = self.mlp(self.post_attn_norm(hidden_states, out_sharded=True))
        return ttnn.add(hidden_states, mlp_out, memory_config=residual_cfg)

    def _residual_memory_config(self, hidden_states):
        """The next norm's decode shard, or None off the decode path.

        Both norms in the block are built on the same dim and so read the same
        shard, and so does the next layer's -- the residual stream can stay in
        it the whole way down. Returns None for prefill, whose many-row norm is
        not sharded at all, so `ttnn.add` keeps its default.
        """
        config = self.post_attn_norm.decode_input_memory_config
        if config is None or not self.post_attn_norm._is_decode_row(hidden_states):
            return None
        return config


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)


# Legacy slug-named shim.
def decoder_layer(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)
