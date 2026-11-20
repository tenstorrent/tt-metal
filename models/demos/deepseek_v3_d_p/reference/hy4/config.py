# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview reference config, read straight from the checkpoint's ``config.json`` (``model_type: hy_v4``).

Only the fields the reference forward and the golden layout need. Normalization mirrors HF
``HYV4Config.__post_init__`` (transformers >= 5.23): ``num_key_value_heads`` is forced to the head
count, ``qk_head_dim = qk_nope + qk_rope``, and absent layer maps fall back to HF's defaults (one dense
layer; a full indexer at layer 0 and every 4th layer from 1).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Hy4RefConfig:
    vocab_size: int = 120832
    hidden_size: int = 6144
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    indexer_types: tuple = field(default_factory=tuple)
    mlp_layer_types: tuple = field(default_factory=tuple)
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0
    rms_norm_eps: float = 1e-5
    # HF builds q_a_layernorm / kv_a_layernorm as HYV4RMSNorm(dim) with its default eps=1e-6, NOT
    # rms_norm_eps (modeling_hy_v4.py). Not a config.json field: HF never reads it from there.
    low_rank_norm_eps: float = 1e-6
    rope_theta: float = 1e7
    bos_token_id: int = 120000
    eos_token_id: int = 120025
    num_nextn_predict_layers: int = 1

    def __post_init__(self):
        n = self.num_hidden_layers
        if not self.indexer_types:
            object.__setattr__(
                self, "indexer_types", tuple("full" if i == 0 or (i - 1) % 4 == 0 else "shared" for i in range(n))
            )
        if not self.mlp_layer_types:
            object.__setattr__(self, "mlp_layer_types", tuple(["dense"] * min(1, n) + ["sparse"] * max(0, n - 1)))
        object.__setattr__(self, "indexer_types", tuple(self.indexer_types))
        object.__setattr__(self, "mlp_layer_types", tuple(self.mlp_layer_types))
        assert len(self.indexer_types) >= n and len(self.mlp_layer_types) >= n, "layer maps shorter than the stack"
        assert self.indexer_types[0] == "full", "layer 0 must own an indexer (a shared layer reuses a prior full one)"
        assert self.index_head_dim > self.qk_rope_head_dim and self.qk_rope_head_dim % 2 == 0

    # ---- derived ----
    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kvpe_dim(self) -> int:
        """Width of the MLA cache line the device writes: kv_a_layernorm(latent) | roped k_pe."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        # rope_type "default": no YaRN mscale (HF yarn_apply_mscale is a no-op).
        return self.qk_head_dim**-0.5

    @property
    def full_indexer_layers(self) -> list[int]:
        """Layers that run their own indexer and own an index-key cache slot (21 of 78 for Hy4-preview)."""
        return [i for i in range(self.num_hidden_layers) if self.indexer_types[i] == "full"]

    def is_full_indexer(self, layer: int) -> bool:
        return self.indexer_types[layer] == "full"

    def is_dense(self, layer: int) -> bool:
        return self.mlp_layer_types[layer] == "dense"

    def truncated(self, num_layers: int) -> "Hy4RefConfig":
        d = asdict(self)
        d["num_hidden_layers"] = num_layers
        d["indexer_types"] = tuple(self.indexer_types[:num_layers])
        d["mlp_layer_types"] = tuple(self.mlp_layer_types[:num_layers])
        return Hy4RefConfig(**d)

    # ---- I/O ----
    @classmethod
    def from_dict(cls, raw: dict) -> "Hy4RefConfig":
        if raw.get("model_type", "hy_v4") != "hy_v4":
            raise ValueError(f"expected model_type hy_v4, got {raw.get('model_type')!r}")
        rope = raw.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(f"only default RoPE is implemented (Hy4-preview); got {rope}")
        names = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in raw.items() if k in names}
        if "rope_theta" in rope:
            kw["rope_theta"] = float(rope["rope_theta"])
        for flag, want in (("use_mla", True), ("use_dsa", True), ("gated_mla", True), ("learnable_sink", True)):
            if raw.get(flag, want) != want:
                raise NotImplementedError(f"{flag}={raw.get(flag)} is not the Hy4-preview architecture")
        if raw.get("gating_type", "elementwise") != "elementwise":
            raise NotImplementedError(f"gating_type={raw.get('gating_type')}")
        return cls(**kw)

    @classmethod
    def from_pretrained(cls, model_dir) -> "Hy4RefConfig":
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))

    def to_hf_dict(self) -> dict:
        """A ``config.json`` HF ``HYV4Config`` loads (used to save the random smoke-test checkpoint)."""
        d = asdict(self)
        d.pop("rope_theta")
        d.pop("low_rank_norm_eps")  # HF hard-codes it (HYV4RMSNorm default), not a config field
        d["indexer_types"] = list(self.indexer_types)
        d["mlp_layer_types"] = list(self.mlp_layer_types)
        d.update(
            model_type="hy_v4",
            architectures=["HYV4ForCausalLM"],
            num_key_value_heads=self.num_attention_heads,
            head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            layer_types=["deepseek_sparse_attention"] * self.num_hidden_layers,
            rope_parameters={"rope_type": "default", "rope_theta": self.rope_theta},
            max_position_embeddings=1048576,
            hidden_act="silu",
            attention_bias=False,
            tie_word_embeddings=False,
            use_mla=True,
            use_dsa=True,
            gated_mla=True,
            gating_type="elementwise",
            learnable_sink=True,
            enable_ihc=True,
            pad_token_id=None,
            dtype="float32",
        )
        return d


def tiny_random_config(**overrides) -> Hy4RefConfig:
    """A few-layer Hy4 with every architectural feature on (iHC, gate, sink, full/shared indexers, dense
    layer 0, clamped MoE, shared expert) and dims small enough for a CPU unit test. ``index_topk`` is
    far below the test sequence length so the sparse selection is exercised."""
    base = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=16,
        v_head_dim=32,
        intermediate_size=96,
        moe_intermediate_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        routed_scaling_factor=2.827,
        swiglu_limit=1.0,  # low so the clamp actually binds at test activations
        index_n_heads=8,
        index_head_dim=32,
        index_topk=16,
        indexer_types=("full", "full", "shared", "shared", "shared", "full"),
        mlp_layer_types=("dense",) + ("sparse",) * 5,
        rope_theta=1e4,
        bos_token_id=1,
        eos_token_id=2,
    )
    base.update(overrides)
    return Hy4RefConfig(**base)
