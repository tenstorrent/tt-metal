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
from ttml.common.profiler_utils import profiler_marker
from ttml.modules import AbstractModuleBase, LinearLayer

from .transformer import RMSNormLayer
from .autograd_ops import autograd_slice, rope_trailing


class MultiHeadLatentAttention(AbstractModuleBase):
    """Multi-head Latent Attention (MLA) layer for DeepSeek-V3.

    Naive-mode flow (no KV-cache absorption):
      Q   : x -> [wq | wq_a -> q_norm -> wq_b]                       # [B, 1, S, H*qk_head]
      KV  : x -> wkv_a -> [kv_latent, k_pe]                          # split low-rank latent + rope-half
            kv_latent -> kv_norm -> wkv_b                            # [B, 1, S, H*(qk_nope_dim+v_dim)]
            k_pe      -> RoPE                                        # [B, 1, S, qk_rope_dim]
      QKV : (q_pre, kv_up, k_pe) -> mla.qkv_assemble                 # head-split + broadcast k_pe + demux v
                                  -> q  [B, H, S, qk_head]           (not yet RoPE'd)
                                  -> k  [B, H, S, qk_head]           (k_nope | broadcast k_pe)
                                  -> v  [B, H, S, v_dim]
            q   -> rope_trailing(rotate trailing qk_rope_dim cols)
      Attn: composite_SDPA(q, k, v, mask) -> heads_fusion -> wo
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
        qk_nope_dim = self.qk_nope_head_dim
        qk_rope_dim = self.qk_rope_head_dim
        v_dim = self.v_head_dim
        kv_latent_rank = self.kv_lora_rank

        x_in = profiler_marker(x, "[START] [MLA]")

        # ── Q projection (head-split is deferred to the fused QKV-assemble below) ──
        x_q = profiler_marker(x_in, "[START] [MLA] Q-Proj")
        if self.q_lora_rank == 0:
            assert self.wq is not None, "Q projection without LoRA requires wq"
            q_pre = self.wq(x_q)
        else:
            assert self.wq_a is not None, "Q projection with LoRA requires wq_a"
            assert self.q_norm is not None, "Q projection with LoRA requires q_norm"
            assert self.wq_b is not None, "Q projection with LoRA requires wq_b"
            q_pre = self.wq_b(self.q_norm(self.wq_a(x_q)))
        # q_pre : [B, 1, S, n_heads * qk_head]
        q_pre = profiler_marker(q_pre, "[END] [MLA] Q-Proj")

        # ── KV down-projection: split into low-rank latent and the shared rope-half ──
        x_kv = profiler_marker(x_in, "[START] [MLA] KV-Down")
        kv_down = self.wkv_a(x_kv)  # [B, 1, S, kv_latent_rank + qk_rope_dim]
        kv_latent = autograd_slice(kv_down, [0, 0, 0, 0], [B, 1, S, kv_latent_rank])
        k_pe = autograd_slice(kv_down, [0, 0, 0, kv_latent_rank], [B, 1, S, kv_latent_rank + qk_rope_dim])
        k_pe = profiler_marker(k_pe, "[END] [MLA] KV-Down")

        # ── RoPE on shared k_pe (per-token, not per-head) ──
        k_pe = profiler_marker(k_pe, "[START] [MLA] RoPE-K")
        k_pe = ttml.ops.rope.rope(k_pe, self.rope_params)  # [B, 1, S, qk_rope_dim]
        k_pe = profiler_marker(k_pe, "[END] [MLA] RoPE-K")

        # ── KV up-projection (produces packed per-head [k_nope | v]) ──
        kv_latent = profiler_marker(kv_latent, "[START] [MLA] KV-Up")
        kv_up = self.wkv_b(self.kv_norm(kv_latent))  # [B, 1, S, n_heads * (qk_nope_dim + v_dim)]
        kv_up = profiler_marker(kv_up, "[END] [MLA] KV-Up")

        # ── Fused QKV assembly ──
        # Q head-split, KV demux into per-head (k_nope, v), and broadcast k_pe
        # into every head's rope-suffix of K. Q is emitted head-split but NOT
        # yet rotated; rope_trailing below applies RoPE to Q's rope-suffix.
        q_pre = profiler_marker(q_pre, "[START] [MLA] QKV-Assemble")
        q, k_full, v = ttml.ops.mla.qkv_assemble(q_pre, kv_up, k_pe, n_heads, qk_nope_dim, qk_rope_dim, v_dim)
        # q, k_full : [B, n_heads, S, qk_head]
        # v         : [B, n_heads, S, v_dim]
        q = profiler_marker(q, "[END] [MLA] QKV-Assemble")

        q = profiler_marker(q, "[START] [MLA] RoPE-Q")
        q = rope_trailing(q, self.rope_params)
        q = profiler_marker(q, "[END] [MLA] RoPE-Q")

        # ── Attention (composite path supports v_dim != qk_head) ──
        q = profiler_marker(q, "[START] [MLA] SDPA")
        attn = ttml.ops.attention.scaled_dot_product_attention_composite(q, k_full, v, mask)
        attn = profiler_marker(attn, "[END] [MLA] SDPA")

        # ── Output projection ──
        attn = profiler_marker(attn, "[START] [MLA] Output")
        attn = ttml.ops.multi_head_utils.heads_fusion(attn)  # [B, 1, S, n_heads * v_dim]
        out = self.wo(attn)
        out = profiler_marker(out, "[END] [MLA] Output")

        return profiler_marker(out, "[END] [MLA]")
