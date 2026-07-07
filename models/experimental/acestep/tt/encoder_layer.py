# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 encoder layer (TTTv2-pattern composition module).

Reference: AceStepEncoderLayer in modeling_acestep_v15_base.py — a standard pre-norm
transformer encoder block (bidirectional, no cache, no AdaLN):

    residual = x
    x = input_layernorm(x)
    x = self_attn(x)              # full or sliding, with RoPE + qk-norm
    x = residual + x
    residual = x
    x = post_attention_layernorm(x)
    x = mlp(x)
    x = residual + x

Pure composition of already-validated building blocks:
  - RMSNorm1D          (TTTv2 reuse)  x2
  - AceStepAttention   (our module)   self-attention, full or sliding
  - MLP1D              (TTTv2 reuse)  SwiGLU
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.experimental.acestep.tt.attention import AceStepAttention, AceStepAttentionConfig


@dataclass
class AceStepEncoderLayerConfig:
    # Norms.
    input_layernorm_weight: LazyWeight
    post_attention_layernorm_weight: LazyWeight

    # Attention (weights pre-transposed to [in,out]).
    wq: LazyWeight
    wk: LazyWeight
    wv: LazyWeight
    wo: LazyWeight
    q_norm_weight: LazyWeight
    k_norm_weight: LazyWeight

    # MLP (w1=gate, w2=down, w3=up, pre-transposed).
    w1: LazyWeight
    w2: LazyWeight
    w3: LazyWeight

    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    eps: float = 1e-6
    sliding_window: int | None = None
    # Optional matmul compute-kernel config. None -> attention/MLP use their own defaults (HiFi2).
    # The LM planner passes HiFi4+fp32-acc here (its fp32 reference rewards higher precision, unlike
    # the DiT-vs-bf16-reference regime); the text encoder leaves it None (unchanged HiFi2).
    compute_kernel_config: object | None = None


class AceStepEncoderLayer(LightweightModule):
    """forward(hidden [1,1,seq,hidden], cos, sin, attn_mask=None) -> [1,1,seq,hidden]."""

    def __init__(self, config: AceStepEncoderLayerConfig):
        self.config = config
        cfg = config

        self.input_layernorm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=cfg.input_layernorm_weight, eps=cfg.eps, compute_kernel_config=cfg.compute_kernel_config
            )
        )
        self.post_attention_layernorm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=cfg.post_attention_layernorm_weight,
                eps=cfg.eps,
                compute_kernel_config=cfg.compute_kernel_config,
            )
        )
        self.self_attn = AceStepAttention(
            AceStepAttentionConfig(
                wq=cfg.wq,
                wk=cfg.wk,
                wv=cfg.wv,
                wo=cfg.wo,
                q_norm_weight=cfg.q_norm_weight,
                k_norm_weight=cfg.k_norm_weight,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                eps=cfg.eps,
                is_cross_attention=False,
                sliding_window=cfg.sliding_window,
                compute_kernel_config=cfg.compute_kernel_config,
            )
        )
        if cfg.compute_kernel_config is not None:
            self.mlp = MLP1D.from_config(
                MLP1DConfig(
                    w1=cfg.w1,
                    w2=cfg.w2,
                    w3=cfg.w3,
                    ff1_3_compute_kernel_cfg=cfg.compute_kernel_config,
                    ff2_compute_kernel_cfg=cfg.compute_kernel_config,
                    decode_ff1_3_compute_kernel_cfg=cfg.compute_kernel_config,
                    decode_ff2_compute_kernel_cfg=cfg.compute_kernel_config,
                )
            )
        else:
            self.mlp = MLP1D(w1=cfg.w1, w2=cfg.w2, w3=cfg.w3)

    @classmethod
    def from_config(cls, config: AceStepEncoderLayerConfig):
        return cls(config)

    def forward(self, hidden_states, cos, sin, attn_mask=None):
        residual = hidden_states
        x = self.input_layernorm.forward(hidden_states, mode="prefill")
        x = self.self_attn.forward(x, cos=cos, sin=sin, attn_mask=attn_mask)
        hidden_states = ttnn.add(residual, x)

        residual = hidden_states
        x = self.post_attention_layernorm.forward(hidden_states, mode="prefill")
        x = self.mlp.forward(x, mode="prefill")
        hidden_states = ttnn.add(residual, x)
        return hidden_states
