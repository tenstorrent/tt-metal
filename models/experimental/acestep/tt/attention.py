# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 attention (TTTv2-style custom module, single-device Blackhole p150).

Why a custom class instead of TTTv2 Attention1D?
  - Attention1D targets *causal autoregressive* LLM decode/prefill with a KV cache and
    hardcodes `is_causal=True` + Meta-convention RoPE (rotary_embedding_llama).
  - ACE-Step's DiT/encoder attention is *bidirectional* (diffusion, no causal mask, no KV
    cache) and uses HF-convention RoPE (rotary_embedding_hf). It also supports a cross-
    attention mode (kv from encoder_hidden_states, no RoPE).

So we follow the TTTv2 *pattern* (LightweightModule + Config dataclass + from_config +
straight-line forward, no static if-else in the hot path) and reuse ttnn ops + RMSNorm1D
for the per-head q/k RMSNorm (Qwen3 qk-norm).

Reference: AceStepAttention in modeling_acestep_v15_base.py (Qwen3-style GQA).
Config: hidden 2048, heads 16, kv_heads 8, head_dim 128, qk-norm, sliding_window=128 on
sliding layers / None on full layers, rope_theta 1e6, no bias, scale = head_dim**-0.5.

Prefill-only (diffusion inference does a single bidirectional forward, no decode/KV cache).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig


@dataclass
class AceStepAttentionConfig:
    """Single source of truth for AceStepAttention. Weights are LazyWeight (HF [in,out] layout)."""

    # Required projection weights (already transposed to [in, out] for ttnn.linear).
    wq: LazyWeight
    wk: LazyWeight
    wv: LazyWeight
    wo: LazyWeight

    # Per-head q/k RMSNorm weights (Qwen3 qk-norm over head_dim). Required for ACE-Step.
    q_norm_weight: LazyWeight | None = None
    k_norm_weight: LazyWeight | None = None

    # Dims.
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    eps: float = 1e-6

    # Behaviour.
    is_cross_attention: bool = False  # kv from encoder_hidden_states, no RoPE
    sliding_window: int | None = None  # window size for sliding layers, None = full attention
    scale: float | None = None  # default head_dim ** -0.5

    mesh_device: ttnn.MeshDevice | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def resolved(self) -> "AceStepAttentionConfig":
        if self.scale is None:
            self.scale = self.head_dim**-0.5
        if self.mesh_device is None:
            self.mesh_device = self.wq.device
        if self.compute_kernel_config is None:
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        return self


class AceStepAttention(LightweightModule):
    """Bidirectional GQA attention with per-head q/k RMSNorm and optional RoPE / cross-attn.

    forward(hidden_states, cos, sin, encoder_hidden_states=None, attn_mask=None) -> ttnn.Tensor
      - hidden_states:        [1, 1, seq, hidden]
      - cos/sin:              [1, 1, seq, head_dim] (self-attn only; ignored for cross-attn)
      - encoder_hidden_states:[1, 1, kv_seq, hidden] (cross-attn only)
      - attn_mask:            optional [1, 1, seq, kv_seq] additive mask (e.g. sliding/padding)
    """

    def __init__(self, config: AceStepAttentionConfig):
        self.config = config.resolved()
        cfg = self.config
        self.wq = cfg.wq.get_device_weight()
        self.wk = cfg.wk.get_device_weight()
        self.wv = cfg.wv.get_device_weight()
        self.wo = cfg.wo.get_device_weight()

        self.q_norm = (
            RMSNorm1D.from_config(RMSNorm1DConfig(weight=cfg.q_norm_weight, eps=cfg.eps))
            if cfg.q_norm_weight is not None
            else None
        )
        self.k_norm = (
            RMSNorm1D.from_config(RMSNorm1DConfig(weight=cfg.k_norm_weight, eps=cfg.eps))
            if cfg.k_norm_weight is not None
            else None
        )

        # Self-attention: fuse Q/K/V into one weight so the head split can use
        # ttnn.experimental.nlp_create_qkv_heads (heads land on dim 1 in one op), avoiding the
        # per-projection reshape+transpose that profiling showed dominates attention (~0.12ms each
        # vs 0.05ms for the matmul). wq/wk/wv are [in, out]; concat on the out axis (dim -1).
        self._fused_qkv = None
        if not cfg.is_cross_attention:
            self._fused_qkv = ttnn.concat([self.wq, self.wk, self.wv], dim=-1)

    @classmethod
    def from_config(cls, config: AceStepAttentionConfig):
        return cls(config)

    def _project_heads(self, x, weight, n_heads):
        """x:[1,1,seq,hidden] -> proj -> [1, n_heads, seq, head_dim]."""
        cfg = self.config
        proj = ttnn.linear(x, weight, compute_kernel_config=cfg.compute_kernel_config)  # [1,1,seq,n*hd]
        seq = proj.shape[2]
        proj = ttnn.reshape(proj, (1, seq, n_heads, cfg.head_dim))  # [1,seq,n,hd]
        proj = ttnn.transpose(proj, 1, 2)  # [1,n,seq,hd]
        return proj

    def forward(self, hidden_states, cos=None, sin=None, encoder_hidden_states=None, attn_mask=None):
        cfg = self.config

        if self._fused_qkv is not None:
            # Self-attention: one fused QKV matmul + nlp_create_qkv_heads (no per-proj transpose).
            fused = ttnn.linear(hidden_states, self._fused_qkv, compute_kernel_config=cfg.compute_kernel_config)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                fused,
                num_heads=cfg.n_heads,
                num_kv_heads=cfg.n_kv_heads,
                transpose_k_heads=False,  # SDPA wants K as [1,nkv,seq,hd], not transposed
            )
        else:
            # Cross-attention: Q from hidden, K/V from encoder (different sources -> keep split path).
            q = self._project_heads(hidden_states, self.wq, cfg.n_heads)  # [1,nq,seq,hd]
            kv_source = encoder_hidden_states
            k = self._project_heads(kv_source, self.wk, cfg.n_kv_heads)  # [1,nkv,kv,hd]
            v = self._project_heads(kv_source, self.wv, cfg.n_kv_heads)  # [1,nkv,kv,hd]

        # Per-head q/k RMSNorm over head_dim (Qwen3 qk-norm), applied on [1,n,seq,hd].
        if self.q_norm is not None:
            q = self.q_norm.forward(q, mode="prefill")
        if self.k_norm is not None:
            k = self.k_norm.forward(k, mode="prefill")

        # RoPE (self-attention only; cross-attention has no positional rotation).
        if not cfg.is_cross_attention and cos is not None:
            q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)

        # Bidirectional SDPA with GQA (nq != nkv handled by the kernel).
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=cfg.scale,
            compute_kernel_config=cfg.compute_kernel_config,
        )  # [1,nq,seq,hd]

        # Concat heads back to [1,1,seq,hidden].
        attn = ttnn.transpose(attn, 1, 2)  # [1,seq,nq,hd]
        seq = attn.shape[1]
        attn = ttnn.reshape(attn, (1, 1, seq, cfg.n_heads * cfg.head_dim))
        out = ttnn.linear(attn, self.wo, compute_kernel_config=cfg.compute_kernel_config)  # [1,1,seq,hidden]
        return out
