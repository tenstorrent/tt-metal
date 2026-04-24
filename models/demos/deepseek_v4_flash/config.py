# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ROOT_TO_INFERENCE_FIELDS = {
    "vocab_size": "vocab_size",
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_hash_layers": "n_hash_layers",
    "num_attention_heads": "n_heads",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "routed_scaling_factor": "route_scale",
    "swiglu_limit": "swiglu_limit",
    "q_lora_rank": "q_lora_rank",
    "head_dim": "head_dim",
    "qk_rope_head_dim": "rope_head_dim",
    "o_groups": "o_groups",
    "o_lora_rank": "o_lora_rank",
    "sliding_window": "window_size",
    "rope_theta": "rope_theta",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "compress_rope_theta": "compress_rope_theta",
    "compress_ratios": "compress_ratios",
}

_ROPE_SCALING_TO_INFERENCE_FIELDS = {
    "original_max_position_embeddings": "original_seq_len",
    "factor": "rope_factor",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
}


@dataclass(frozen=True)
class DeepSeekV4FlashConfig:
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_nextn_predict_layers: int
    num_hash_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    q_lora_rank: int
    qk_rope_head_dim: int
    o_groups: int
    o_lora_rank: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: str
    routed_scaling_factor: float
    swiglu_limit: float
    sliding_window: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    rms_norm_eps: float
    rope_theta: float
    compress_rope_theta: float
    compress_ratios: tuple[int, ...]
    rope_scaling: dict[str, Any]
    quantization_config: dict[str, Any]
    torch_dtype: str
    model_type: str = "deepseek_v4"
    expert_dtype: str | None = None

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> "DeepSeekV4FlashConfig":
        model_path = Path(model_path)
        root_config = _read_json_object(model_path / "config.json")
        inference_path = model_path / "inference" / "config.json"
        inference_config = _read_json_object(inference_path) if inference_path.is_file() else None
        return cls.from_hf_configs(root_config, inference_config)

    @classmethod
    def from_hf_configs(
        cls, root_config: dict[str, Any], inference_config: dict[str, Any] | None = None
    ) -> "DeepSeekV4FlashConfig":
        _require(root_config.get("model_type") == "deepseek_v4", "Expected config.model_type to be 'deepseek_v4'")
        _require(
            root_config.get("architectures") in (None, ["DeepseekV4ForCausalLM"]),
            "Expected DeepseekV4ForCausalLM architecture",
        )
        compress_ratios = tuple(int(value) for value in _expect_list(root_config, "compress_ratios"))
        num_hidden_layers = _expect_int(root_config, "num_hidden_layers")
        num_nextn_predict_layers = int(root_config.get("num_nextn_predict_layers", 0))
        expected_ratio_lengths = {num_hidden_layers, num_hidden_layers + num_nextn_predict_layers}
        _require(
            len(compress_ratios) in expected_ratio_lengths,
            f"compress_ratios length {len(compress_ratios)} must match num_hidden_layers {num_hidden_layers} "
            f"or num_hidden_layers + num_nextn_predict_layers {num_hidden_layers + num_nextn_predict_layers}",
        )

        if inference_config is not None:
            mismatches = compare_root_and_inference_config(root_config, inference_config)
            _require(not mismatches, "Root config and inference/config.json disagree: " + "; ".join(mismatches))

        quantization_config = _expect_dict(root_config, "quantization_config")
        weight_block_size = quantization_config.get("weight_block_size")
        _require(weight_block_size == [128, 128], f"Expected FP8 weight_block_size [128, 128], got {weight_block_size}")

        expert_dtype = None
        if inference_config is not None:
            expert_dtype = inference_config.get("expert_dtype")

        return cls(
            vocab_size=_expect_int(root_config, "vocab_size"),
            hidden_size=_expect_int(root_config, "hidden_size"),
            moe_intermediate_size=_expect_int(root_config, "moe_intermediate_size"),
            num_hidden_layers=num_hidden_layers,
            num_nextn_predict_layers=num_nextn_predict_layers,
            num_hash_layers=_expect_int(root_config, "num_hash_layers"),
            num_attention_heads=_expect_int(root_config, "num_attention_heads"),
            num_key_value_heads=_expect_int(root_config, "num_key_value_heads"),
            head_dim=_expect_int(root_config, "head_dim"),
            q_lora_rank=_expect_int(root_config, "q_lora_rank"),
            qk_rope_head_dim=_expect_int(root_config, "qk_rope_head_dim"),
            o_groups=_expect_int(root_config, "o_groups"),
            o_lora_rank=_expect_int(root_config, "o_lora_rank"),
            n_routed_experts=_expect_int(root_config, "n_routed_experts"),
            n_shared_experts=_expect_int(root_config, "n_shared_experts"),
            num_experts_per_tok=_expect_int(root_config, "num_experts_per_tok"),
            scoring_func=_expect_str(root_config, "scoring_func"),
            routed_scaling_factor=_expect_float(root_config, "routed_scaling_factor"),
            swiglu_limit=_expect_float(root_config, "swiglu_limit"),
            sliding_window=_expect_int(root_config, "sliding_window"),
            index_n_heads=_expect_int(root_config, "index_n_heads"),
            index_head_dim=_expect_int(root_config, "index_head_dim"),
            index_topk=_expect_int(root_config, "index_topk"),
            hc_mult=_expect_int(root_config, "hc_mult"),
            hc_sinkhorn_iters=_expect_int(root_config, "hc_sinkhorn_iters"),
            hc_eps=_expect_float(root_config, "hc_eps"),
            rms_norm_eps=_expect_float(root_config, "rms_norm_eps"),
            rope_theta=_expect_float(root_config, "rope_theta"),
            compress_rope_theta=_expect_float(root_config, "compress_rope_theta"),
            compress_ratios=compress_ratios,
            rope_scaling=dict(_expect_dict(root_config, "rope_scaling")),
            quantization_config=dict(quantization_config),
            torch_dtype=_expect_str(root_config, "torch_dtype"),
            model_type=_expect_str(root_config, "model_type"),
            expert_dtype=expert_dtype,
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_nextn_predict_layers": self.num_nextn_predict_layers,
            "num_hash_layers": self.num_hash_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "o_groups": self.o_groups,
            "o_lora_rank": self.o_lora_rank,
            "n_routed_experts": self.n_routed_experts,
            "n_shared_experts": self.n_shared_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "scoring_func": self.scoring_func,
            "routed_scaling_factor": self.routed_scaling_factor,
            "swiglu_limit": self.swiglu_limit,
            "sliding_window": self.sliding_window,
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
            "hc_mult": self.hc_mult,
            "hc_sinkhorn_iters": self.hc_sinkhorn_iters,
            "hc_eps": self.hc_eps,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "compress_rope_theta": self.compress_rope_theta,
            "compress_ratios": list(self.compress_ratios),
            "rope_scaling": self.rope_scaling,
            "quantization_config": self.quantization_config,
            "torch_dtype": self.torch_dtype,
            "expert_dtype": self.expert_dtype,
        }


def compare_root_and_inference_config(root_config: dict[str, Any], inference_config: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for root_field, inference_field in _ROOT_TO_INFERENCE_FIELDS.items():
        if root_field not in root_config or inference_field not in inference_config:
            continue
        root_value = root_config[root_field]
        inference_value = inference_config[inference_field]
        if root_value != inference_value:
            mismatches.append(f"{root_field}={root_value!r} vs {inference_field}={inference_value!r}")

    rope_scaling = root_config.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        for root_field, inference_field in _ROPE_SCALING_TO_INFERENCE_FIELDS.items():
            if root_field not in rope_scaling or inference_field not in inference_config:
                continue
            root_value = rope_scaling[root_field]
            inference_value = inference_config[inference_field]
            if root_value != inference_value:
                mismatches.append(f"rope_scaling.{root_field}={root_value!r} vs {inference_field}={inference_value!r}")
    return mismatches


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    _require(isinstance(obj, dict), f"Expected JSON object in {path}")
    return obj


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expect_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    _require(isinstance(value, int), f"Expected integer config field '{key}', got {type(value).__name__}")
    return value


def _expect_float(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    _require(isinstance(value, (int, float)), f"Expected numeric config field '{key}', got {type(value).__name__}")
    return float(value)


def _expect_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    _require(isinstance(value, str), f"Expected string config field '{key}', got {type(value).__name__}")
    return value


def _expect_dict(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    _require(isinstance(value, dict), f"Expected object config field '{key}', got {type(value).__name__}")
    return value


def _expect_list(config: dict[str, Any], key: str) -> list[Any]:
    value = config.get(key)
    _require(isinstance(value, list), f"Expected list config field '{key}', got {type(value).__name__}")
    return value
