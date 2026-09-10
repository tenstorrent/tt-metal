# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model configuration parsed from the raw Hugging Face ``config.json`` (no transformers dependency)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig

KDA = "linear_attention"
MLA = "full_attention"
DENSE = "dense"
MOE = "sparse"


@dataclass(frozen=True)
class KimiLinearConfig:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    max_position_embeddings: int
    # KDA
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_size: int
    kda_layers_1idx: tuple[int, ...]
    full_attn_layers_1idx: tuple[int, ...]
    # MLA (NoPE)
    num_attention_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: int | None
    mla_use_nope: bool
    # MoE / MLP
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_token: int
    num_shared_experts: int
    first_k_dense_replace: int
    moe_layer_freq: int
    moe_renormalize: bool
    moe_router_activation_func: str
    routed_scaling_factor: float
    num_expert_group: int
    topk_group: int
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # ---- derived -----------------------------------------------------------------------------
    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim  # 192

    @property
    def kv_latent_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim  # 576 = cached latent width

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(KDA if (i + 1) in self.kda_layers_1idx else MLA for i in range(self.num_hidden_layers))

    @property
    def mlp_layer_types(self) -> tuple[str, ...]:
        return tuple(
            DENSE
            if (i < self.first_k_dense_replace or (i - self.first_k_dense_replace) % self.moe_layer_freq != 0)
            else MOE
            for i in range(self.num_hidden_layers)
        )

    def is_kda_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == KDA

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == MOE

    def kda_config(self) -> KDAConfig:
        return KDAConfig(
            hidden_size=self.hidden_size,
            num_heads=self.linear_num_heads,
            head_k_dim=self.linear_head_dim,
            head_v_dim=self.linear_head_dim,
            conv_kernel_size=self.linear_conv_kernel_size,
            norm_eps=self.rms_norm_eps,
            use_full_rank_gate=bool(self.raw.get("linear_attn_config", {}).get("use_full_rank_gate", False)),
            gate_lower_bound=self.raw.get("linear_attn_config", {}).get("gate_lower_bound"),
        )

    @property
    def mla_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i))

    @property
    def kda_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.is_kda_layer(i))

    # ---- construction ----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "KimiLinearConfig":
        lin = d["linear_attn_config"]
        eos = d.get("eos_token_id", 163586)
        if isinstance(eos, list):
            eos = eos[0]
        return cls(
            hidden_size=int(d["hidden_size"]),
            num_hidden_layers=int(d["num_hidden_layers"]),
            vocab_size=int(d["vocab_size"]),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-5)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
            bos_token_id=int(d.get("bos_token_id", 163584)),
            eos_token_id=int(eos),
            pad_token_id=int(d.get("pad_token_id", 163839)),
            max_position_embeddings=int(d.get("model_max_length", d.get("max_position_embeddings", 1048576))),
            linear_num_heads=int(lin["num_heads"]),
            linear_head_dim=int(lin["head_dim"]),
            linear_conv_kernel_size=int(lin["short_conv_kernel_size"]),
            kda_layers_1idx=tuple(int(x) for x in lin["kda_layers"]),
            full_attn_layers_1idx=tuple(int(x) for x in lin["full_attn_layers"]),
            num_attention_heads=int(d["num_attention_heads"]),
            kv_lora_rank=int(d["kv_lora_rank"]),
            qk_nope_head_dim=int(d["qk_nope_head_dim"]),
            qk_rope_head_dim=int(d["qk_rope_head_dim"]),
            v_head_dim=int(d["v_head_dim"]),
            q_lora_rank=(int(d["q_lora_rank"]) if d.get("q_lora_rank") is not None else None),
            mla_use_nope=bool(d.get("mla_use_nope", True)),
            intermediate_size=int(d["intermediate_size"]),
            moe_intermediate_size=int(d["moe_intermediate_size"]),
            num_experts=int(d.get("num_experts", d.get("n_routed_experts", d.get("num_local_experts")))),
            num_experts_per_token=int(d.get("num_experts_per_token", d.get("num_experts_per_tok"))),
            num_shared_experts=int(d.get("num_shared_experts", d.get("n_shared_experts", 1)) or 0),
            first_k_dense_replace=int(d.get("first_k_dense_replace", 0)),
            moe_layer_freq=int(d.get("moe_layer_freq", 1)),
            moe_renormalize=bool(d.get("moe_renormalize", d.get("norm_topk_prob", True))),
            moe_router_activation_func=str(d.get("moe_router_activation_func", "sigmoid")),
            routed_scaling_factor=float(d.get("routed_scaling_factor", 1.0)),
            num_expert_group=int(d.get("num_expert_group", d.get("n_group", 1))),
            topk_group=int(d.get("topk_group", 1)),
            raw=dict(d),
        )

    @classmethod
    def from_snapshot(cls, snapshot_dir: str | Path) -> "KimiLinearConfig":
        return cls.from_dict(json.load(open(Path(snapshot_dir) / "config.json")))

    def validate(self) -> None:
        assert set(self.kda_layers_1idx) | set(self.full_attn_layers_1idx) == set(range(1, self.num_hidden_layers + 1))
        assert self.q_lora_rank is None, "this port assumes a direct q_proj (q_lora_rank null)"
        assert self.mla_use_nope, "this port assumes NoPE MLA layers"
        assert (
            self.num_expert_group == 1 and self.topk_group == 1
        ), "grouped top-k not implemented (identity for 1 group)"
        assert self.moe_router_activation_func == "sigmoid"
        assert self.linear_conv_kernel_size == 4
