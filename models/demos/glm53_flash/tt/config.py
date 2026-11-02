# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model configuration for GLM-5.3-Flash (``Glm5NextForConditionalGeneration``, text tower).

All values come from the checkpoint's ``config.json``; nothing here is tunable. The device-side
precision policy and parallel layout live in the Phase-2 mesh module and the module program configs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KDA = "linear_attention"
DSA = "deepseek_sparse_attention"


@dataclass(frozen=True)
class GLM53FlashConfig:
    hidden_size: int = 4096
    vocab_size: int = 154880
    num_hidden_layers: int = 45
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    # mHC hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # schedule
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    mlp_layer_types: tuple[str, ...] = field(default_factory=tuple)
    # KDA (Kimi delta attention)
    kda_num_heads: int = 64
    kda_head_dim: int = 128
    kda_conv_kernel: int = 4
    kda_gate_lower_bound: float = -5.0
    # DSA / MLA
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    # DSA indexer
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    # MLP / MoE
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    swiglu_limit: float = 10.0
    # tokens
    pad_token_id: int = 154820
    eos_token_ids: tuple[int, ...] = (154820, 154827, 154829)
    image_token_id: int = 154854

    # ---- derived -------------------------------------------------------------------------
    @property
    def kda_qkv_dim(self) -> int:
        return self.kda_num_heads * self.kda_head_dim  # 8192

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim  # 256

    @property
    def mla_softmax_scale(self) -> float:
        return self.qk_head_dim**-0.5

    @property
    def index_softmax_scale(self) -> float:
        return self.index_head_dim**-0.5

    @property
    def index_select_pools(self) -> int:
        return self.index_topk // self.index_kpool  # 512

    @property
    def index_max_selected_tokens(self) -> int:
        """Selected token ids per query: topk pools expanded plus the incomplete tail pool."""
        return self.index_topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)  # 2051

    def layer_kind(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def is_kda(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == KDA

    def is_moe(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "sparse"

    @property
    def kda_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == KDA)

    @property
    def dsa_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t != KDA)

    # ---- construction ----------------------------------------------------------------------
    @classmethod
    def from_hf_config_json(cls, path: str | Path) -> "GLM53FlashConfig":
        cfg = json.loads(Path(path).read_text())
        t = cfg["text_config"]
        la = t["linear_attn_config"]
        layer_types = tuple(KDA if lt == KDA else DSA for lt in t["layer_types"])
        return cls(
            hidden_size=t["hidden_size"],
            vocab_size=t["vocab_size"],
            num_hidden_layers=t["num_hidden_layers"],
            rms_norm_eps=t["rms_norm_eps"],
            max_position_embeddings=t["max_position_embeddings"],
            hc_mult=t["hc_mult"],
            hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
            hc_eps=t["hc_eps"],
            layer_types=layer_types,
            mlp_layer_types=tuple(t["mlp_layer_types"]),
            kda_num_heads=la["num_heads"],
            kda_head_dim=la["head_dim"],
            kda_conv_kernel=la["short_conv_kernel_size"],
            kda_gate_lower_bound=la["gate_lower_bound"],
            num_attention_heads=t["num_attention_heads"],
            q_lora_rank=t["q_lora_rank"],
            kv_lora_rank=t["kv_lora_rank"],
            qk_nope_head_dim=t["qk_nope_head_dim"],
            qk_rope_head_dim=t["qk_rope_head_dim"],
            v_head_dim=t["v_head_dim"],
            index_n_heads=t["index_n_heads"],
            index_head_dim=t["index_head_dim"],
            index_topk=t["index_topk"],
            index_kpool=t["index_kpool"],
            index_kpool_always_select_tail=t["index_kpool_always_select_tail"],
            intermediate_size=t["intermediate_size"],
            moe_intermediate_size=t["moe_intermediate_size"],
            n_routed_experts=t["n_routed_experts"],
            n_shared_experts=t["n_shared_experts"],
            num_experts_per_tok=t["num_experts_per_tok"],
            routed_scaling_factor=t["routed_scaling_factor"],
            norm_topk_prob=t["norm_topk_prob"],
            scoring_func=t["scoring_func"],
            swiglu_limit=t["swiglu_limit"],
            pad_token_id=t["pad_token_id"],
            eos_token_ids=tuple(t["eos_token_id"]),
            image_token_id=cfg["image_token_id"],
        )

    @classmethod
    def from_snapshot(cls, snapshot: str | Path | None = None) -> "GLM53FlashConfig":
        if snapshot is None:
            from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir

            snapshot = default_snapshot_dir()
        return cls.from_hf_config_json(Path(snapshot) / "config.json")

    def __post_init__(self):
        if self.layer_types:
            assert len(self.layer_types) == self.num_hidden_layers
            assert len(self.mlp_layer_types) == self.num_hidden_layers
        assert self.hc_mult * self.hc_mult <= 32, "mhc_split_sinkhorn packs n*n into one tile row"
        assert self.qk_rope_head_dim == 0, "GLM-5.3-Flash is NoPE; the MLA path assumes no rope part"
        assert self.index_topk % self.index_kpool == 0
        assert self.scoring_func == "sigmoid" and self.n_shared_experts == 1
