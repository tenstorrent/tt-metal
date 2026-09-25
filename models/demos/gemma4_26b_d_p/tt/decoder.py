# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 decoder layer: input_norm -> attention -> post_attn_norm -> +residual -> FFN (incl. residual, scalar)."""

import ttnn
from models.demos.gemma4_26b_d_p.reference.config import Gemma4TextConfig
from models.demos.gemma4_26b_d_p.tt.attention.attention import TtAttention
from models.demos.gemma4_26b_d_p.tt.ffn import TtFFN, TtRMSNorm


class TtDecoderLayer:
    def __init__(self, mesh_device, cfg: Gemma4TextConfig, layer_idx: int, layer_sd: dict, *, ccl, sp_topology, tp_topology,
                 seq_len_per_chip, expert_dtype=ttnn.bfloat8_b, num_links=1):
        self.layer_idx = layer_idx
        self.is_sliding = cfg.is_sliding(layer_idx)
        attn_sd = {k[len("self_attn.") :]: v for k, v in layer_sd.items() if k.startswith("self_attn.")}
        self.input_norm = TtRMSNorm(mesh_device, layer_sd["input_layernorm.weight"], cfg.rms_norm_eps)
        self.post_attn_norm = TtRMSNorm(mesh_device, layer_sd["post_attention_layernorm.weight"], cfg.rms_norm_eps)
        self.attn = TtAttention(mesh_device, cfg, layer_idx, attn_sd, ccl, tp_topology)
        self.ffn = TtFFN(mesh_device, cfg, layer_sd, seq_len_per_chip=seq_len_per_chip, sp_topology=sp_topology, num_links=num_links,
                         expert_dtype=expert_dtype, layer_idx=layer_idx)

    def __call__(self, x, rope, trans_mat, kv_cache, *, cache_layer, kv_actual, user=0, valid_end=None):
        h = self.input_norm(x)
        a = self.attn(h, rope, trans_mat, kv_cache, cache_layer=cache_layer, kv_actual=kv_actual, user=user, valid_end=valid_end)
        h.deallocate(True)
        an = self.post_attn_norm(a)
        a.deallocate(True)
        r = ttnn.add(x, an)
        an.deallocate(True)
        out = self.ffn(r)
        r.deallocate(True)
        return out
