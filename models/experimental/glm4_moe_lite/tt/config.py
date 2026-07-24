# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Glm4MoeLiteHParams:
    # Core
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int

    # Attention (DeepSeek-style multi-latent attention)
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_head_dim: int

    # Norm / RoPE
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    rope_interleave: bool

    # MoE
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    topk_method: str

    @staticmethod
    def _rope_theta_from_hf(hf_config: Any) -> float:
        # Prefer explicit rope_theta if present; otherwise fall back to rope_parameters.
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is not None:
            return float(rope_theta)
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            return float(rope_parameters["rope_theta"])
        raise ValueError("Unable to determine rope_theta from hf_config (no rope_theta or rope_parameters.rope_theta)")

    @staticmethod
    def _partial_rotary_factor_from_hf(hf_config: Any) -> float:
        prf = getattr(hf_config, "partial_rotary_factor", None)
        if prf is not None:
            return float(prf)
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "partial_rotary_factor" in rope_parameters:
            return float(rope_parameters["partial_rotary_factor"])
        return 1.0

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "Glm4MoeLiteHParams":
        qk_nope_head_dim = int(getattr(hf_config, "qk_nope_head_dim"))
        qk_rope_head_dim = int(getattr(hf_config, "qk_rope_head_dim"))
        qk_head_dim = int(getattr(hf_config, "qk_head_dim", qk_nope_head_dim + qk_rope_head_dim))

        return cls(
            vocab_size=int(getattr(hf_config, "vocab_size")),
            hidden_size=int(getattr(hf_config, "hidden_size")),
            intermediate_size=int(getattr(hf_config, "intermediate_size")),
            num_hidden_layers=int(getattr(hf_config, "num_hidden_layers")),
            num_attention_heads=int(getattr(hf_config, "num_attention_heads")),
            num_key_value_heads=int(
                getattr(hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads"))
            ),
            q_lora_rank=int(getattr(hf_config, "q_lora_rank")),
            kv_lora_rank=int(getattr(hf_config, "kv_lora_rank")),
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=int(getattr(hf_config, "v_head_dim")),
            qk_head_dim=qk_head_dim,
            rms_norm_eps=float(getattr(hf_config, "rms_norm_eps")),
            rope_theta=cls._rope_theta_from_hf(hf_config),
            partial_rotary_factor=cls._partial_rotary_factor_from_hf(hf_config),
            rope_interleave=bool(getattr(hf_config, "rope_interleave", True)),
            moe_intermediate_size=int(getattr(hf_config, "moe_intermediate_size")),
            n_routed_experts=int(getattr(hf_config, "n_routed_experts")),
            n_shared_experts=int(getattr(hf_config, "n_shared_experts")),
            num_experts_per_tok=int(getattr(hf_config, "num_experts_per_tok")),
            first_k_dense_replace=int(getattr(hf_config, "first_k_dense_replace", 0)),
            norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
            routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
            n_group=int(getattr(hf_config, "n_group", 1)),
            topk_group=int(getattr(hf_config, "topk_group", 1)),
            topk_method=str(getattr(hf_config, "topk_method", "noaux_tc")),
        )

    def validate(self) -> None:
        assert self.vocab_size > 0
        assert self.hidden_size > 0
        assert self.intermediate_size > 0
        assert self.num_hidden_layers > 0
        assert self.num_attention_heads > 0
        assert self.num_key_value_heads > 0
        assert self.q_lora_rank > 0
        assert self.kv_lora_rank > 0
        assert self.qk_nope_head_dim > 0
        assert self.qk_rope_head_dim > 0
        assert self.v_head_dim > 0
        assert self.qk_head_dim == (self.qk_nope_head_dim + self.qk_rope_head_dim)
        assert self.rms_norm_eps > 0
        assert self.rope_theta > 0
        assert 0 < self.partial_rotary_factor <= 1.0
        assert self.moe_intermediate_size > 0
        assert self.n_routed_experts > 0
        assert self.n_shared_experts >= 0
        assert self.num_experts_per_tok > 0
        assert self.first_k_dense_replace >= 0
        assert isinstance(self.norm_topk_prob, bool)
        assert self.routed_scaling_factor > 0
        assert self.n_group > 0
        assert 0 < self.topk_group <= self.n_group
        assert self.topk_method
