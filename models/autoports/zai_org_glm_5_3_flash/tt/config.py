# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (`glm5_next_text`) configuration for the TTNN functional decoder.

`Glm5NextTextConfig` parses the checkpoint's `config.json` (nested `text_config`) without importing transformers; field
semantics follow upstream `configuration_glm5_next.py` (transformers e24eb546fefd) including the legacy layer-type remap
`deepseek_sparse_attention -> indexed_attention`. The module adds the device-side constants the decoder needs; it never
shrinks a model dimension.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

_LAYER_TYPE_REMAP = {
    "deepseek_sparse_attention": "indexed_attention",  # configuration_utils.py:96 (checkpoint spelling)
    "full_attention": "indexed_attention",  # configuration_glm5_next.py:170-172
}


@dataclass
class Glm5NextTextConfig:
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    pad_token_id: int | None = 154820
    eos_token_id: list[int] = field(default_factory=lambda: [154820, 154827, 154829])
    attention_bias: bool = False
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    swiglu_limit: float = 10.0
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    num_nextn_predict_layers: int = 1
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    indexer_types: list[str] | None = None

    def __post_init__(self):
        n = self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, n) + ["sparse"] * (n - 3)
        if self.layer_types is None:
            self.layer_types = ["linear_attention" if i % 4 != 3 else "indexed_attention" for i in range(n)]
        self.layer_types = [_LAYER_TYPE_REMAP.get(t, t) for t in self.layer_types]
        if self.indexer_types is None:
            self.indexer_types = ["full"] * n
        assert set(self.layer_types) <= {"linear_attention", "indexed_attention"}, self.layer_types
        assert set(self.mlp_layer_types) <= {"dense", "sparse"}, self.mlp_layer_types
        assert self.qk_rope_head_dim == 0, "GLM-5.3-Flash attention is NoPE"
        assert self.num_attention_heads == self.num_key_value_heads
        assert self.index_topk % self.index_kpool == 0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def num_local_experts(self) -> int:
        return self.n_routed_experts

    @classmethod
    def from_json(cls, path: str, **overrides) -> "Glm5NextTextConfig":
        """Load from the snapshot's `config.json` (nested `text_config`) or a flat text config file."""
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as f:
            raw = json.load(f)
        tc = dict(raw.get("text_config", raw))
        # configuration_glm5_next.py:189-197: linear_attn_config overrides the flat linear_* fields
        lac = tc.pop("linear_attn_config", None)
        if lac is not None:
            tc["linear_head_dim"] = lac.get("head_dim", tc.get("linear_head_dim", 128))
            tc["linear_num_heads"] = lac.get("num_heads", tc.get("linear_num_heads", 64))
            tc["linear_conv_kernel_dim"] = lac.get("short_conv_kernel_size", tc.get("linear_conv_kernel_dim", 4))
            tc["linear_lower_bound"] = lac.get("gate_lower_bound", tc.get("linear_lower_bound", -5.0))
            if lac.get("safe_gate", True) and tc["linear_lower_bound"] is None:
                tc["linear_lower_bound"] = -5.0
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in tc.items() if k in known}
        kwargs.update(overrides)
        return cls(**kwargs)

    def layer_kind(self, layer_idx: int) -> str:
        """One of 'kda_dense', 'kda_moe', 'mla_moe', 'mla_dense' (the latter does not occur in this checkpoint)."""
        attn = "kda" if self.layer_types[layer_idx] == "linear_attention" else "mla"
        mlp = "moe" if self.mlp_layer_types[layer_idx] == "sparse" else "dense"
        return f"{attn}_{mlp}"


TILE = 32

# Layer kinds exercised by the functional decoder (names used in evidence rows).
DENSE_KDA = "dense_kda"  # layers 0-2: KDA attention + dense SwiGLU MLP
SPARSE_KDA = "sparse_kda"  # 31 layers: KDA attention + MoE
SPARSE_DSA = "sparse_dsa"  # 11 layers: MLA/DSA attention + MoE

# DSA top-k width: index_topk (2048) + kpool-1 (3) tail tokens. For kv_len <= 2051 every visible token is selected, so
# dense causal MLA attention is exactly the HF computation (modeling_glm5_next.py:883-913).
DSA_DENSE_EXACT_MAX_KV = 2048 + 3


def load_hf_text_config(snapshot_dir: str) -> Glm5NextTextConfig:
    """Parse the HF snapshot's `config.json` (nested `text_config`)."""
    path = os.path.join(snapshot_dir, "config.json") if os.path.isdir(snapshot_dir) else snapshot_dir
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no HF config.json at {snapshot_dir}")
    return Glm5NextTextConfig.from_json(path)


@dataclass(frozen=True)
class LayerSpec:
    """Static description of one decoder layer, derived only from the HF config."""

    layer_idx: int
    attention: str  # "kda" | "mla"
    mlp: str  # "dense" | "moe"

    @property
    def kind(self) -> str:
        if self.attention == "kda":
            return DENSE_KDA if self.mlp == "dense" else SPARSE_KDA
        if self.mlp == "moe":
            return SPARSE_DSA
        raise ValueError(f"layer {self.layer_idx}: MLA + dense MLP does not occur in GLM-5.3-Flash")

    @classmethod
    def from_config(cls, cfg: Glm5NextTextConfig, layer_idx: int) -> "LayerSpec":
        if not 0 <= layer_idx < cfg.num_hidden_layers:
            raise ValueError(
                f"layer_idx {layer_idx} outside [0, {cfg.num_hidden_layers}) (MTP layer 45 is out of scope)"
            )
        attention = "kda" if cfg.layer_types[layer_idx] == "linear_attention" else "mla"
        mlp = "moe" if cfg.mlp_layer_types[layer_idx] == "sparse" else "dense"
        if cfg.indexer_types[layer_idx] != "full":
            raise ValueError(
                f"layer {layer_idx}: shared-indexer DSA layers are not supported (none in this checkpoint)"
            )
        return cls(layer_idx=layer_idx, attention=attention, mlp=mlp)


def round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m
