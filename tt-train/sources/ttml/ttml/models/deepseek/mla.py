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
from .autograd_ops import autograd_slice, autograd_concat, split_heads


class MultiHeadLatentAttention(AbstractModuleBase):
    """Multi-head Latent Attention (MLA) layer.

    Follows the DeepSeek-V3 naive-mode attention:
      Q (q_lora_rank > 0): x -> wq_a -> norm -> wq_b -> split_heads
      Q (q_lora_rank == 0): x -> wq -> split_heads   (direct, no LoRA bottleneck)
      Both: -> [q_nope, q_pe] -> RoPE(q_pe) -> cat
      KV: x -> wkv_a -> [kv_latent, k_pe] -> norm(kv_latent) -> wkv_b -> split_heads
          -> [k_nope, v] + RoPE(k_pe) broadcast -> cat(k_nope, k_pe)
      Attention: composite_SDPA(Q, K, V, mask) -> fuse_heads -> wo
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

    def forward(self, x: ttml.autograd.Tensor, mask: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        B, _, S, _ = list(x.get_value().shape)
        n_heads = self.n_heads
        qk_nope = self.qk_nope_head_dim
        qk_rope = self.qk_rope_head_dim
        qk_head = self.qk_head_dim
        v_dim = self.v_head_dim

        # ── Q path ──
        if self.q_lora_rank == 0:
            q = self.wq(x)  # [B, 1, S, n_heads * qk_head]
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))  # [B, 1, S, n_heads * qk_head]
        q = split_heads(q, n_heads)  # [B, n_heads, S, qk_head]

        q_nope = autograd_slice(q, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
        q_pe = autograd_slice(q, [0, 0, 0, qk_nope], [B, n_heads, S, qk_head])
        q_pe = ttml.ops.rope.rope(q_pe, self.rope_params)

        # ── KV path ──
        kv_full = self.wkv_a(x)  # [B, 1, S, kv_lora_rank + qk_rope]
        kv_lora = self.kv_lora_rank

        kv = autograd_slice(kv_full, [0, 0, 0, 0], [B, 1, S, kv_lora])
        k_pe = autograd_slice(kv_full, [0, 0, 0, kv_lora], [B, 1, S, kv_lora + qk_rope])

        # RoPE on k_pe (shared across heads, shape [B, 1, S, qk_rope])
        k_pe = ttml.ops.rope.rope(k_pe, self.rope_params)

        # Expand k_pe to all heads: [B, 1, S, qk_rope] -> [B, n_heads, S, qk_rope]
        k_pe = autograd_concat([k_pe] * n_heads, dim=1)

        # Up-project KV latent and split into per-head k_nope and v
        kv_up = self.wkv_b(self.kv_norm(kv))  # [B, 1, S, n_heads * (qk_nope + v_dim)]
        kv_up = split_heads(kv_up, n_heads)  # [B, n_heads, S, qk_nope + v_dim]

        k_nope = autograd_slice(kv_up, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
        v = autograd_slice(kv_up, [0, 0, 0, qk_nope], [B, n_heads, S, qk_nope + v_dim])

        # Assemble full Q and K
        q_full = autograd_concat([q_nope, q_pe], dim=3)  # [B, H, S, qk_head]
        k_full = autograd_concat([k_nope, k_pe], dim=3)  # [B, H, S, qk_head]

        # ── Attention (composite path supports v_dim != qk_head) ──
        attn = ttml.ops.attention.scaled_dot_product_attention_composite(q_full, k_full, v, mask)

        # ── Output ──
        attn = ttml.ops.multi_head_utils.heads_fusion(attn)  # [B, 1, S, n_heads * v_dim]
        return self.wo(attn)
