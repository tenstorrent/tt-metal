# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 decoder layer: x + attn(input_norm(x)) -> r; r + ffn(post_attn_norm(r))."""

import ttnn
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.tt.attention.attention import TtAttention
from models.demos.mimo_v2_d_p.tt.ccl import default_num_links
from models.demos.mimo_v2_d_p.tt.ffn import TtDenseMLP, TtMoE, TtRMSNorm


class TtDecoderLayer:
    def __init__(self, mesh_device, cfg: MiMoTextConfig, layer_idx: int, layer_sd: dict, *, ccl, sp_topology, seq_len_per_chip,
                 expert_dtype=None, num_links=None):
        self.layer_idx = layer_idx
        num_links = num_links or default_num_links()
        self.kind = cfg.layer_type(layer_idx)
        sub = lambda p: {k[len(p) + 1 :]: v for k, v in layer_sd.items() if k.startswith(p + ".")}
        eps = cfg.layernorm_epsilon
        self.input_norm = TtRMSNorm(mesh_device, layer_sd["input_layernorm.weight"], eps)
        self.post_attn_norm = TtRMSNorm(mesh_device, layer_sd["post_attention_layernorm.weight"], eps)
        cp = f"L{layer_idx}"
        self.attn = TtAttention(mesh_device, cfg, layer_idx, sub("self_attn"), ccl, cache_prefix=cp)
        if cfg.is_moe(layer_idx):
            self.ffn = TtMoE(mesh_device, sub("mlp"), cfg, seq_len_per_chip=seq_len_per_chip, num_links=num_links, topology=sp_topology,
                             weights_dtype=expert_dtype, cache_prefix=cp)
        else:
            self.ffn = TtDenseMLP(mesh_device, sub("mlp"), cache_prefix=cp)

    def __call__(self, x, rope, trans_mat, kv_cache, *, cache_layer, kv_actual, user=0, valid_end=None):
        h = self.input_norm(x)
        a = self.attn(h, rope, trans_mat, kv_cache, cache_layer=cache_layer, kv_actual=kv_actual, user=user, valid_end=valid_end)
        h.deallocate(True)
        r = ttnn.add(x, a)
        a.deallocate(True)
        h = self.post_attn_norm(r)
        f = self.ffn(h)
        h.deallocate(True)
        out = ttnn.add(r, f)
        r.deallocate(True)
        f.deallocate(True)
        return out
