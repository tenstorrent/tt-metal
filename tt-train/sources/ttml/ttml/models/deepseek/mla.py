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
from .autograd_ops import autograd_split


class MultiHeadLatentAttention(AbstractModuleBase):
    """Multi-head Latent Attention (MLA) layer.

    Follows the DeepSeek-V3 naive-mode attention:
      Q (q_lora_rank > 0): x -> wq_a -> norm -> wq_b
      Q (q_lora_rank == 0): x -> wq (direct, no LoRA bottleneck)
      KV: x -> wkv_a -> split(kv_latent, k_pe) -> norm(kv_latent) -> wkv_b
          -> RoPE(k_pe) broadcast
      Q/K/V assembly: qkv_assemble + mla_q_rope on Q
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
        n_heads = self.n_heads
        qk_nope = self.qk_nope_head_dim
        qk_rope = self.qk_rope_head_dim
        v_dim = self.v_head_dim

        # ── Q path ──
        if self.q_lora_rank == 0:
            q_pre = self.wq(x)  # [B, 1, S, n_heads * qk_head]
        else:
            q_pre = self.wq_b(self.q_norm(self.wq_a(x)))  # [B, 1, S, n_heads * qk_head]

        # ── KV path ──
        kv_full = self.wkv_a(x)  # [B, 1, S, kv_lora_rank + qk_rope]
        kv, k_pe = autograd_split(kv_full, [self.kv_lora_rank, qk_rope], dim=3)

        # RoPE on k_pe (shared across heads, shape [B, 1, S, qk_rope])
        k_pe = ttml.ops.rope.rope(k_pe, self.rope_params)

        kv_up = self.wkv_b(self.kv_norm(kv))  # [B, 1, S, n_heads * (qk_nope + v_dim)]

        q, k_full, v = ttml.ops.mla.qkv_assemble(q_pre, kv_up, k_pe, n_heads, qk_nope, qk_rope, v_dim)
        q_full = ttml.ops.rope.mla_q_rope(q, self.rope_params, qk_nope, qk_rope)

        # ── Attention (causal-only) ──
        # None -> fused SDPA generates the causal mask on chip and takes the faster
        # causal/balanced path (vs a materialized arbitrary mask). MLA is causal-only,
        # so there is deliberately no mask argument; see the class docstring.
        attn = ttml.ops.attention.scaled_dot_product_attention(q_full, k_full, v, None)

        # ── Output ──
        attn = ttml.ops.multi_head_utils.heads_fusion(attn)  # [B, 1, S, n_heads * v_dim]
        return self.wo(attn)
