# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2.6-Flash text config, parsed from the HF ``config.json`` without importing remote code."""

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionDims:
    is_swa: bool
    num_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    rope_dim: int
    rope_theta: float
    sliding_window: int | None
    has_sink: bool

    @property
    def scaling(self):
        return self.head_dim**-0.5


class MiMoConfig:
    def __init__(self, hf_config: dict):
        c = hf_config
        self.raw = c
        self.hidden_size = c["hidden_size"]
        self.num_hidden_layers = c["num_hidden_layers"]
        self.vocab_size = c["vocab_size"]
        self.layernorm_epsilon = c["layernorm_epsilon"]
        self.hybrid_layer_pattern = list(c["hybrid_layer_pattern"])
        self.moe_layer_freq = list(c["moe_layer_freq"])
        self.intermediate_size = c["intermediate_size"]
        self.moe_intermediate_size = c["moe_intermediate_size"]
        self.n_routed_experts = c["n_routed_experts"]
        self.num_experts_per_tok = c["num_experts_per_tok"]
        self.norm_topk_prob = c["norm_topk_prob"]
        self.routed_scaling_factor = c.get("routed_scaling_factor") or 1.0
        assert c["scoring_func"] == "sigmoid" and c["topk_method"] == "noaux_tc"
        assert c["n_group"] == 1 and c["topk_group"] == 1
        self.num_attention_heads = c["num_attention_heads"]
        self.num_key_value_heads = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.v_head_dim = c["v_head_dim"]
        self.swa_num_attention_heads = c.get("swa_num_attention_heads", self.num_attention_heads)
        self.swa_num_key_value_heads = c.get("swa_num_key_value_heads", self.num_key_value_heads)
        self.swa_head_dim = c.get("swa_head_dim", self.head_dim)
        self.swa_v_head_dim = c.get("swa_v_head_dim", self.v_head_dim)
        self.partial_rotary_factor = c.get("partial_rotary_factor", 1.0)
        self.rope_theta = c["rope_theta"]
        self.swa_rope_theta = c.get("swa_rope_theta", self.rope_theta)
        self.sliding_window = c["sliding_window"]
        self.attention_value_scale = c.get("attention_value_scale")
        self.add_swa_attention_sink_bias = c.get("add_swa_attention_sink_bias", False)
        self.add_full_attention_sink_bias = c.get("add_full_attention_sink_bias", False)
        self.max_position_embeddings = c["max_position_embeddings"]
        self.eos_token_id = c.get("eos_token_id")
        # MTP layers (model.mtp.layers.*): SWA attention with sinks and a dense MLP, addressed by the layer indices
        # after the decoder's (mtp_layer_index).
        self.num_nextn_predict_layers = c.get("num_nextn_predict_layers", 0)

    @classmethod
    def from_model_path(cls, model_path):
        with open(os.path.join(model_path, "config.json")) as f:
            return cls(json.load(f))

    def mtp_layer_index(self, i):
        """The layer index of MTP layer ``i``: is_swa, is_moe and attention_dims take it like a decoder layer's."""
        assert 0 <= i < self.num_nextn_predict_layers, f"MTP layer {i} of {self.num_nextn_predict_layers}"
        return self.num_hidden_layers + i

    def is_mtp(self, layer_idx):
        return layer_idx >= self.num_hidden_layers

    def is_swa(self, layer_idx):
        return self.is_mtp(layer_idx) or self.hybrid_layer_pattern[layer_idx] == 1

    def is_moe(self, layer_idx):
        return not self.is_mtp(layer_idx) and bool(self.moe_layer_freq[layer_idx])

    def attention_dims(self, layer_idx) -> AttentionDims:
        swa = self.is_swa(layer_idx)
        head_dim = self.swa_head_dim if swa else self.head_dim
        rope_dim = int(head_dim * self.partial_rotary_factor)
        assert rope_dim % 2 == 0
        return AttentionDims(
            is_swa=swa,
            num_heads=self.swa_num_attention_heads if swa else self.num_attention_heads,
            num_kv_heads=self.swa_num_key_value_heads if swa else self.num_key_value_heads,
            head_dim=head_dim,
            v_head_dim=self.swa_v_head_dim if swa else self.v_head_dim,
            rope_dim=rope_dim,
            rope_theta=self.swa_rope_theta if swa else self.rope_theta,
            sliding_window=self.sliding_window if swa else None,
            has_sink=self.add_swa_attention_sink_bias if swa else self.add_full_attention_sink_bias,
        )
