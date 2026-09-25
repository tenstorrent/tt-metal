# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash text-backbone config (the parts a prefill layer needs).

Hybrid attention: 9 global-attention (GA) layers (``hybrid_layer_pattern == 0``) and 39 sliding-window
(SWA, window 128, per-head attention sink) layers. Both share 64 Q heads, qk head_dim 192 (first 64 dims
NeoX-rotated) and v head_dim 128; they differ in KV heads (GA 4, SWA 8) and rope theta (GA 1e7, SWA 1e4).
Layer 0 has a dense SwiGLU MLP (16384); every other layer is a 256-expert top-8 sigmoid MoE (2048).
V is scaled by ``attention_value_scale`` before attention (folded into the V projection here).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

GA, SWA = "full_attention", "sliding_attention"


@dataclass(frozen=True)
class AttnSpec:
    kind: str
    n_q: int
    n_kv: int
    head_dim: int  # qk
    v_head_dim: int
    rope_dim: int
    rope_theta: float
    window: int | None
    has_sink: bool


@dataclass
class MiMoTextConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    swa_num_key_value_heads: int = 8
    head_dim: int = 192
    v_head_dim: int = 128
    partial_rotary_factor: float = 0.334
    rope_theta: float = 1e7
    swa_rope_theta: float = 1e4
    sliding_window: int = 128
    attention_value_scale: float = 0.707
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    layernorm_epsilon: float = 1e-6
    vocab_size: int = 152576
    hybrid_layer_pattern: list = field(default_factory=lambda: [0] + ([1] * 4 + [0]) * 1 + ([1] * 5 + [0]) * 7)
    moe_layer_freq: list = field(default_factory=lambda: [0] + [1] * 47)

    @classmethod
    def from_json(cls, path=None):
        if path is None or not Path(path).exists():
            return cls()
        d = json.loads(Path(path).read_text())
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d and d[k] is not None}
        return cls(**kw)

    def layer_type(self, i):
        return SWA if self.hybrid_layer_pattern[i] == 1 else GA

    def is_moe(self, i):
        return bool(self.moe_layer_freq[i])

    def attn_spec(self, kind: str) -> AttnSpec:
        swa = kind == SWA
        rope_dim = int(self.head_dim * self.partial_rotary_factor)
        return AttnSpec(
            kind=kind,
            n_q=self.num_attention_heads,
            n_kv=self.swa_num_key_value_heads if swa else self.num_key_value_heads,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            rope_dim=rope_dim,
            rope_theta=self.swa_rope_theta if swa else self.rope_theta,
            window=self.sliding_window if swa else None,
            has_sink=self.add_swa_attention_sink_bias if swa else self.add_full_attention_sink_bias,
        )

    def layer_attn(self, i) -> AttnSpec:
        return self.attn_spec(self.layer_type(i))
