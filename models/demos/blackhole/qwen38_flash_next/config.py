# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed configuration and four-device placement for Qwen3.8-Flash-Next."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
LAYER_PATTERN = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 12


def _expect(mapping: dict[str, Any], name: str, expected: Any) -> Any:
    actual = mapping.get(name)
    if actual != expected:
        raise ValueError(f"config field {name} must be {expected!r}, got {actual!r}")
    return actual


@dataclass(frozen=True)
class Qwen38Config:
    checkpoint_root: Path
    config_sha256: str
    hidden_size: int
    residual_branches: int
    residual_rank: int
    vocab_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    rms_norm_eps: float
    max_position_embeddings: int
    hidden_act: str
    gdn_qk_heads: int
    gdn_value_heads: int
    gdn_key_head_dim: int
    gdn_value_head_dim: int
    gdn_conv_kernel: int
    gdn_output_gate: str
    qsa_query_heads: int
    qsa_kv_heads: int
    qsa_head_dim: int
    qsa_rope_dim: int
    index_query_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_budget: int
    index_compress_ratio: int
    num_experts: int
    top_k: int
    norm_topk_prob: bool
    expert_intermediate_size: int
    shared_expert_intermediate_size: int
    ple_checkpoint_layer: int
    ple_embedding_width: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ngram_shards: int
    ngram_divisible_by: int
    ple_seed: int
    ple_conv_kernel: int
    mtp_layers: int
    mtp_uses_shared_embeddings: bool
    eos_token_id: int
    rope_theta: int

    @property
    def residual_width(self) -> int:
        return self.residual_branches * self.hidden_size

    @property
    def gdn_qk_width(self) -> int:
        return self.gdn_qk_heads * self.gdn_key_head_dim

    @property
    def gdn_value_width(self) -> int:
        return self.gdn_value_heads * self.gdn_value_head_dim

    @property
    def qsa_query_width(self) -> int:
        return self.qsa_query_heads * self.qsa_head_dim

    @property
    def qsa_kv_width(self) -> int:
        return self.qsa_kv_heads * self.qsa_head_dim

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_root: str | Path,
        *,
        config_file: str | Path | None = None,
    ) -> "Qwen38Config":
        checkpoint_root = Path(checkpoint_root).resolve()
        config_path = Path(config_file) if config_file is not None else checkpoint_root / "config.json"
        raw = config_path.read_bytes()
        document = json.loads(raw)
        text = document.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("config field text_config must be an object")

        _expect(document, "architectures", ["Qwen4ExpForConditionalGeneration"])
        _expect(document, "model_type", "qwen4_exp")
        _expect(document, "language_model_only", False)
        _expect(text, "model_type", "qwen4_exp_text")
        _expect(text, "dtype", "bfloat16")
        _expect(text, "hidden_size", 2560)
        _expect(text, "hc_count", 4)
        _expect(text, "hc_lowrank", 320)
        _expect(text, "vocab_size", 248320)
        _expect(text, "num_hidden_layers", 48)
        _expect(text, "layer_types", list(LAYER_PATTERN))
        _expect(text, "rms_norm_eps", 1e-6)
        _expect(text, "max_position_embeddings", 262144)
        _expect(text, "hidden_act", "silu")
        _expect(text, "linear_num_key_heads", 16)
        _expect(text, "linear_num_value_heads", 48)
        _expect(text, "linear_key_head_dim", 128)
        _expect(text, "linear_value_head_dim", 128)
        _expect(text, "linear_conv_kernel_dim", 4)
        _expect(text, "output_gate_type", "sigmoid")
        _expect(text, "mamba_ssm_dtype", "float32")
        _expect(text, "num_attention_heads", 24)
        _expect(text, "num_key_value_heads", 2)
        _expect(text, "head_dim", 256)
        _expect(text, "partial_rotary_factor", 0.25)
        _expect(text, "indexer_n_heads", 4)
        _expect(text, "indexer_kv_heads", 1)
        _expect(text, "indexer_head_dim", 128)
        _expect(text, "indexer_budget", 2048)
        _expect(text, "indexer_compress_ratio", 4)
        _expect(text, "num_experts", 512)
        _expect(text, "num_experts_per_tok", 10)
        norm_topk_prob = text.get("norm_topk_prob", True)
        if norm_topk_prob is not True:
            raise ValueError(
                f"config field norm_topk_prob must resolve to the Qwen4Exp default True, got {norm_topk_prob!r}"
            )
        _expect(text, "moe_intermediate_size", 640)
        _expect(text, "shared_expert_intermediate_size", 640)
        _expect(text, "ple_layer_ids", [2])
        _expect(text, "ple_embed_dim", 2560)
        _expect(text, "ngram_size", 3)
        _expect(text, "heads_per_ngram", 8)
        _expect(text, "ngram_vocab_size_base", 20_000_000)
        _expect(text, "split_ngram_parts", 128)
        _expect(text, "make_ngram_vocab_size_divisible_by", 128)
        ple_seed = text.get("seed", 1234)
        if ple_seed != 1234:
            raise ValueError(f"config field seed must resolve to the Qwen4Exp default 1234, got {ple_seed!r}")
        _expect(text, "ple_conv_kernel_size", 4)
        _expect(text, "mtp_num_hidden_layers", 1)
        _expect(text, "mtp_use_dedicated_embeddings", False)
        _expect(text, "eos_token_id", 248044)
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict):
            raise ValueError("config field rope_parameters must be an object")
        _expect(rope, "rope_theta", 10_000_000)
        _expect(rope, "partial_rotary_factor", 0.25)

        digest = hashlib.sha256(raw).hexdigest()
        if digest != CONFIG_SHA256:
            raise ValueError(f"config SHA-256 must be {CONFIG_SHA256}, got {digest}")

        return cls(
            checkpoint_root=checkpoint_root,
            config_sha256=digest,
            hidden_size=text["hidden_size"],
            residual_branches=text["hc_count"],
            residual_rank=text["hc_lowrank"],
            vocab_size=text["vocab_size"],
            num_hidden_layers=text["num_hidden_layers"],
            layer_types=tuple(text["layer_types"]),
            rms_norm_eps=text["rms_norm_eps"],
            max_position_embeddings=text["max_position_embeddings"],
            hidden_act=text["hidden_act"],
            gdn_qk_heads=text["linear_num_key_heads"],
            gdn_value_heads=text["linear_num_value_heads"],
            gdn_key_head_dim=text["linear_key_head_dim"],
            gdn_value_head_dim=text["linear_value_head_dim"],
            gdn_conv_kernel=text["linear_conv_kernel_dim"],
            gdn_output_gate=text["output_gate_type"],
            qsa_query_heads=text["num_attention_heads"],
            qsa_kv_heads=text["num_key_value_heads"],
            qsa_head_dim=text["head_dim"],
            qsa_rope_dim=int(text["head_dim"] * text["partial_rotary_factor"]),
            index_query_heads=text["indexer_n_heads"],
            index_kv_heads=text["indexer_kv_heads"],
            index_head_dim=text["indexer_head_dim"],
            index_budget=text["indexer_budget"],
            index_compress_ratio=text["indexer_compress_ratio"],
            num_experts=text["num_experts"],
            top_k=text["num_experts_per_tok"],
            norm_topk_prob=norm_topk_prob,
            expert_intermediate_size=text["moe_intermediate_size"],
            shared_expert_intermediate_size=text["shared_expert_intermediate_size"],
            ple_checkpoint_layer=text["ple_layer_ids"][0] - 1,
            ple_embedding_width=text["ple_embed_dim"],
            ngram_size=text["ngram_size"],
            heads_per_ngram=text["heads_per_ngram"],
            ngram_vocab_size_base=text["ngram_vocab_size_base"],
            ngram_shards=text["split_ngram_parts"],
            ngram_divisible_by=text["make_ngram_vocab_size_divisible_by"],
            ple_seed=ple_seed,
            ple_conv_kernel=text["ple_conv_kernel_size"],
            mtp_layers=text["mtp_num_hidden_layers"],
            mtp_uses_shared_embeddings=not text["mtp_use_dedicated_embeddings"],
            eos_token_id=text["eos_token_id"],
            rope_theta=rope["rope_theta"],
        )


def _ranges(total: int, parts: int) -> tuple[tuple[int, int], ...]:
    if total % parts:
        raise ValueError(f"dimension {total} does not divide over {parts} devices")
    width = total // parts
    return tuple((index * width, (index + 1) * width) for index in range(parts))


@dataclass(frozen=True)
class Qwen38Placement:
    """Numerical TP4/EP4 placement; runtime topology assertions bind it to hardware."""

    config: Qwen38Config
    mesh_shape: tuple[int, int]
    physical_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.physical_ids) != 4:
            raise ValueError(f"Qwen3.8-Flash-Next requires exactly four physical devices, got {len(self.physical_ids)}")
        if tuple(self.mesh_shape) != (1, 4):
            raise ValueError(f"Qwen3.8-Flash-Next requires a 1x4 mesh, got {self.mesh_shape}")
        if len(set(self.physical_ids)) != 4:
            raise ValueError(f"physical IDs must be four distinct devices, got {self.physical_ids}")

    @property
    def hidden_ranges(self) -> tuple[tuple[int, int], ...]:
        return _ranges(self.config.hidden_size, 4)

    @property
    def vocab_ranges(self) -> tuple[tuple[int, int], ...]:
        return _ranges(self.config.vocab_size, 4)

    @property
    def expert_ranges(self) -> tuple[tuple[int, int], ...]:
        return _ranges(self.config.num_experts, 4)

    @property
    def gdn_qk_head_ranges(self) -> tuple[tuple[int, int], ...]:
        return _ranges(self.config.gdn_qk_heads, 4)

    @property
    def gdn_value_head_ranges(self) -> tuple[tuple[int, int], ...]:
        return _ranges(self.config.gdn_value_heads, 4)

    @property
    def qsa_query_head_ranges(self) -> tuple[tuple[int, int], ...]:
        return _ranges(self.config.qsa_query_heads, 4)

    @property
    def qsa_kv_device_groups(self) -> tuple[tuple[int, int], ...]:
        # KV head 0 serves Q heads 0..11 on devices 0/1; KV head 1 serves
        # Q heads 12..23 on devices 2/3.
        return ((0, 1), (2, 3))

    @property
    def index_query_head_ranges(self) -> tuple[tuple[int, int], ...]:
        return _ranges(self.config.index_query_heads, 4)

    @property
    def index_key_replicas(self) -> tuple[int, ...]:
        return (0, 1, 2, 3)

    @property
    def ple_result_ranges(self) -> tuple[tuple[int, int], ...]:
        return self.hidden_ranges

    def classify_tensor(self, name: str) -> str:
        ple_prefix = (
            f"model.language_model.layers.{self.config.ple_checkpoint_layer}.ple."
            "ple_embedding.ngram_embedding.shard_"
        )
        if name.startswith(ple_prefix):
            return "host_ple"
        if ".mlp.experts." in name:
            return "expert_parallel"
        if name in {"model.language_model.embed_tokens.weight", "lm_head.weight"}:
            return "vocab_parallel"
        if ".self_attn.k_proj.weight" in name or ".self_attn.v_proj.weight" in name:
            return "qsa_kv_grouped"
        if ".self_attn.indexer.index_qk_proj.weight" in name:
            return "qsa_index_explicit"
        return "tensor_parallel"
