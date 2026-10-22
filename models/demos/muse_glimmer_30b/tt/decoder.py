# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer sandwich-norm decoder layer (TP, replicated residual stream)."""

import ttnn
from models.demos.muse_glimmer_30b.tt.attention import MuseGlimmerAttention
from models.demos.muse_glimmer_30b.tt.mlp import MuseGlimmerMLP
from models.demos.muse_glimmer_30b.tt.rms_norm import RMSNorm


class MuseGlimmerDecoderLayer:
    """x = x + post_attn_ln(attn(input_ln(x))); x = x + post_ff_ln(mlp(pre_ff_ln(x))).

    Ownership: forward_* never deallocates the input ``x`` (the caller owns it); it returns a NEW tensor
    that the caller owns. All internal intermediates are deallocated here.
    """

    def __init__(self, args, mesh_device, ccl, layer_idx, state_dict=None, use_weight_cache=True):
        self.args = args
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        if state_dict is None:
            state_dict = args.load_layer_state_dict(layer_idx)

        def norm(name, eps):
            return RMSNorm(
                mesh_device,
                state_dict[f"{name}.weight"],
                eps,
                add_unit_offset=True,
                cache_file_name=args.cache_file(f"layers.{layer_idx}.{name}", ttnn.bfloat16, "p2t")
                if use_weight_cache
                else None,
            )

        self.input_layernorm = norm("input_layernorm", args.rms_norm_eps)
        self.post_attention_layernorm = norm("post_attention_layernorm", args.post_norm_eps)
        self.pre_feedforward_layernorm = norm("pre_feedforward_layernorm", args.rms_norm_eps)
        self.post_feedforward_layernorm = norm("post_feedforward_layernorm", args.post_norm_eps)
        self.attn = MuseGlimmerAttention(args, mesh_device, ccl, layer_idx, state_dict, use_weight_cache)
        self.mlp = MuseGlimmerMLP(args, mesh_device, ccl, layer_idx, state_dict, use_weight_cache)

    def _block(self, x, attn_fn):
        h = self.input_layernorm(x)
        a = attn_fn(h)
        if a is not h:
            ttnn.deallocate(h)
        a2 = self.post_attention_layernorm(a)
        ttnn.deallocate(a)
        x1 = ttnn.add(x, a2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a2)

        h = self.pre_feedforward_layernorm(x1)
        m = self.mlp(h)
        ttnn.deallocate(h)
        m2 = self.post_feedforward_layernorm(m)
        ttnn.deallocate(m)
        out = ttnn.add(x1, m2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(m2)
        ttnn.deallocate(x1)
        return out

    def forward_prefill(self, x, rope, kv_cache, page_table, user_id):
        return self._block(x, lambda h: self.attn.forward_prefill(h, rope, kv_cache, page_table, user_id))

    def forward_decode(self, x, current_pos, rope, kv_cache, page_table):
        return self._block(x, lambda h: self.attn.forward_decode(h, current_pos, rope, kv_cache, page_table))
