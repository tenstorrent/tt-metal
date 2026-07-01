# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-head Latent Attention (MLA) for DeepSeek-V3.

Naive mode implementation (no KV cache absorption) suitable for training.
Key features:
  - Low-rank Q projection: wq_a -> RMSNorm -> wq_b
  - Joint low-rank KV compression: wkv_a -> split(kv_latent, k_pe)
  - RoPE applied only to the rope portion of Q/K heads
  - k_pe shared across all heads (broadcast)
  - Composite SDPA (supports v_head_dim != qk_head_dim)
"""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

from .transformer import RMSNormLayer
from .autograd_ops import autograd_slice, autograd_concat, autograd_split, split_heads

composite = True


def kv_down_split(kv_full, B, S, kv_lora, qk_rope):
    kv = autograd_slice(kv_full, [0, 0, 0, 0], [B, 1, S, kv_lora])
    k_pe = autograd_slice(kv_full, [0, 0, 0, kv_lora], [B, 1, S, kv_lora + qk_rope])
    return kv, k_pe


def qkv_assemble(q_pre, kv_up, k_pe, B, n_heads, S, qk_nope, v_head_dim):
    q = split_heads(q_pre, n_heads)
    kv_up_heads = split_heads(kv_up, n_heads)
    k_nope = autograd_slice(kv_up_heads, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
    v = autograd_slice(kv_up_heads, [0, 0, 0, qk_nope], [B, n_heads, S, qk_nope + v_head_dim])
    k_pe_broadcast = autograd_concat([k_pe] * n_heads, dim=1)
    k_full = autograd_concat([k_nope, k_pe_broadcast], dim=3)
    return q, k_full, v


def q_rope(q, rope_params, B, n_heads, S, qk_nope, qk_head):
    q_nope = autograd_slice(q, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
    q_pe = autograd_slice(q, [0, 0, 0, qk_nope], [B, n_heads, S, qk_head])
    q_pe = ttml.ops.rope.rope(q_pe, rope_params)
    return autograd_concat([q_nope, q_pe], dim=3)


class MultiHeadLatentAttention(AbstractModuleBase):
    """Multi-head Latent Attention (MLA) layer.

    Follows the DeepSeek-V3 naive-mode attention:
      Q (q_lora_rank > 0): x -> wq_a -> norm -> wq_b -> split_heads
      Q (q_lora_rank == 0): x -> wq -> split_heads   (direct, no LoRA bottleneck)
      Both: -> [q_nope, q_pe] -> RoPE(q_pe) -> cat
      KV: x -> wkv_a -> [kv_latent, k_pe] -> norm(kv_latent) -> wkv_b -> split_heads
          -> [k_nope, v] + RoPE(k_pe) broadcast -> cat(k_nope, k_pe)
      Attention: fused causal SDPA(Q, K, V) -> fuse_heads -> wo

    Causal-only: the fused SDPA generates the causal mask on chip, so this layer
    takes no mask argument. A non-causal/custom mask would only matter for
    sequence packing or padding, which the DeepSeek training path does not use.
    """

    def __init__(self, config, rope_params) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.rope_params = rope_params

        # Q path: direct projection or LoRA bottleneck
        if config.q_lora_rank == 0:
            self.wq = LinearLayer(config.dim, config.n_heads * self.qk_head_dim, has_bias=False)
            self.wq_a = None
            self.q_norm = None
            self.wq_b = None
        else:
            self.wq = None
            self.wq_a = LinearLayer(config.dim, config.q_lora_rank, has_bias=False)
            self.q_norm = RMSNormLayer(config.q_lora_rank)
            self.wq_b = LinearLayer(config.q_lora_rank, config.n_heads * self.qk_head_dim, has_bias=False)

        # KV path: joint down-project (kv_latent + k_pe)
        self.wkv_a = LinearLayer(config.dim, config.kv_lora_rank + config.qk_rope_head_dim, has_bias=False)
        self.kv_norm = RMSNormLayer(config.kv_lora_rank)
        self.wkv_b = LinearLayer(
            config.kv_lora_rank,
            config.n_heads * (config.qk_nope_head_dim + config.v_head_dim),
            has_bias=False,
        )

        # Output projection
        self.wo = LinearLayer(config.n_heads * config.v_head_dim, config.dim, has_bias=False)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        B, _, S, _ = list(x.get_value().shape)
        n_heads = self.n_heads
        qk_nope = self.qk_nope_head_dim
        qk_rope = self.qk_rope_head_dim
        qk_head = self.qk_head_dim
        v_dim = self.v_head_dim

        # ── Q path ──
        if self.q_lora_rank == 0:
            q_pre = self.wq(x)  # [B, 1, S, n_heads * qk_head]
        else:
            q_pre = self.wq_b(self.q_norm(self.wq_a(x)))  # [B, 1, S, n_heads * qk_head]

        # ── KV path ──
        kv_full = self.wkv_a(x)  # [B, 1, S, kv_lora_rank + qk_rope]
        if composite:
            kv, k_pe = kv_down_split(kv_full, B, S, self.kv_lora_rank, qk_rope)
        else:
            kv, k_pe = autograd_split(kv_full, [self.kv_lora_rank, qk_rope], dim=3)

        # RoPE on k_pe (shared across heads, shape [B, 1, S, qk_rope])
        k_pe = ttml.ops.rope.rope(k_pe, self.rope_params)

        kv_up = self.wkv_b(self.kv_norm(kv))  # [B, 1, S, n_heads * (qk_nope + v_dim)]

        if composite:
            q, k_full, v = qkv_assemble(q_pre, kv_up, k_pe, B, n_heads, S, qk_nope, v_dim)
        else:
            q, k_full, v = ttml.ops.mla.qkv_assemble(q_pre, kv_up, k_pe, n_heads, qk_nope, qk_rope, v_dim)
        if composite:
            q_full = q_rope(q, self.rope_params, B, n_heads, S, qk_nope, qk_head)
        else:
            q_full = ttml.ops.rope.mla_q_rope(q, self.rope_params, qk_nope, qk_rope)

        # ── Attention (causal-only) ──
        # None -> fused SDPA generates the causal mask on chip and takes the faster
        # causal/balanced path (vs a materialized arbitrary mask). MLA is causal-only,
        # so there is deliberately no mask argument; see the class docstring.
        attn = ttml.ops.attention.scaled_dot_product_attention(q_full, k_full, v, None)

        # ── Output ──
        attn = ttml.ops.multi_head_utils.heads_fusion(attn)  # [B, 1, S, n_heads * v_dim]
        return self.wo(attn)
