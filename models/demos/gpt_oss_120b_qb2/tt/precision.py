# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Runtime-consumed precision policy for the GPT-OSS 120B autoport."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import ttnn
from models.demos.gpt_oss_120b_qb2.tt.multichip_decoder import DEFAULT_MULTICHIP_POLICY, MultichipDecoderPolicy

DEFAULT_PRECISION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "precision.json"

_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}
_MATH_FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
_WEIGHT_KEYS = {
    "embedding",
    "attention_qkv",
    "attention_output",
    "router",
    "experts_gate_up",
    "experts_down",
    "normalization",
    "lm_head",
}
_FIDELITY_KEYS = {
    "decode_attention_projection",
    "prefill_attention_projection",
    "attention_sdpa",
    "expert_matmul",
    "router_matmul",
    "lm_head",
}
_ACTIVATION_KEYS = {"stack_residual", "attention_projection_input", "expert_intermediate"}
_CCL_KEYS = {"attention", "experts"}
_LOGITS_KEYS = {
    "lm_head_output",
    "full_logits_gather",
    "topk_values_gather_dtype",
    "sampling_accumulator",
}
_FULL_LOGITS_GATHER_KEYS = {"mode"}
_LAYER_WEIGHT_KEYS = {
    "attention_qkv",
    "attention_output",
    "router",
    "experts_gate_up",
    "experts_down",
    "normalization",
}
_LAYER_FIDELITY_KEYS = _FIDELITY_KEYS - {"lm_head"}
_LAYER_OVERRIDE_KEYS = {
    "weight_groups",
    "compute_fidelities",
    "activation_residual_dtype",
    "ccl_dtype",
    "kv_cache_dtype",
}


def dtype_name(dtype) -> str:
    for name, value in _DTYPES.items():
        if dtype == value:
            return name
    raise ValueError(f"unsupported TTNN dtype: {dtype!r}")


def math_fidelity_name(fidelity) -> str:
    for name, value in _MATH_FIDELITIES.items():
        if fidelity == value:
            return name
    raise ValueError(f"unsupported TTNN math fidelity: {fidelity!r}")


def _require_exact_keys(mapping: dict, expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _validate_named_values(mapping: dict, allowed: dict, label: str) -> None:
    for key, value in mapping.items():
        if value not in allowed:
            raise ValueError(f"{label}.{key} has unsupported value {value!r}; choose from {sorted(allowed)}")


def _merge_policy(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in _LAYER_OVERRIDE_KEYS:
            raise ValueError(f"unsupported layer exception field {key!r}")
        if isinstance(value, dict):
            allowed = {
                "weight_groups": _LAYER_WEIGHT_KEYS,
                "compute_fidelities": _LAYER_FIDELITY_KEYS,
                "activation_residual_dtype": _ACTIVATION_KEYS,
                "ccl_dtype": _CCL_KEYS,
            }[key]
            unknown = sorted(set(value) - allowed)
            if unknown:
                raise ValueError(f"unsupported layer exception {key} keys: {unknown}")
            merged[key].update(value)
        else:
            if key != "kv_cache_dtype":
                raise ValueError(f"layer exception {key} must be an object")
            merged[key] = value
    return merged


@dataclass(frozen=True)
class PrecisionConfig:
    raw: dict
    source_path: Path | None = None

    def __post_init__(self):
        self.validate()

    @property
    def config_id(self) -> str:
        return self.raw["config_id"]

    @property
    def source_sha256(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def _base_policy_dict(self) -> dict:
        return {
            key: copy.deepcopy(value)
            for key, value in self.raw.items()
            if key not in {"schema_version", "config_id", "layer_exceptions"}
        }

    def policy_dict_for_layer(self, layer_idx: int) -> dict:
        override = self.raw["layer_exceptions"].get(str(int(layer_idx)), {})
        return _merge_policy(self._base_policy_dict(), override)

    def decoder_policy_for_layer(self, layer_idx: int) -> MultichipDecoderPolicy:
        policy = self.policy_dict_for_layer(layer_idx)
        weights = policy["weight_groups"]
        fidelities = policy["compute_fidelities"]
        activations = policy["activation_residual_dtype"]
        ccl = policy["ccl_dtype"]
        return replace(
            DEFAULT_MULTICHIP_POLICY,
            name=f"datatype_sweep:{self.config_id}:layer{layer_idx}",
            attention_weight_dtype=_DTYPES[weights["attention_qkv"]],
            expert_weight_dtype=_DTYPES[weights["experts_gate_up"]],
            router_weight_dtype=_DTYPES[weights["router"]],
            normalization_weight_dtype=_DTYPES[weights["normalization"]],
            kv_cache_dtype=_DTYPES[policy["kv_cache_dtype"]],
            attention_activation_ccl_dtype=_DTYPES[ccl["attention"]],
            expert_activation_ccl_dtype=_DTYPES[ccl["experts"]],
            projection_math_fidelity=_MATH_FIDELITIES[fidelities["decode_attention_projection"]],
            prefill_projection_math_fidelity=_MATH_FIDELITIES[fidelities["prefill_attention_projection"]],
            attention_sdpa_math_fidelity=_MATH_FIDELITIES[fidelities["attention_sdpa"]],
            expert_math_fidelity=_MATH_FIDELITIES[fidelities["expert_matmul"]],
            router_math_fidelity=_MATH_FIDELITIES[fidelities["router_matmul"]],
            residual_dtype=_DTYPES[activations["stack_residual"]],
            attention_projection_input_dtype=_DTYPES[activations["attention_projection_input"]],
            expert_intermediate_dtype=_DTYPES[activations["expert_intermediate"]],
        )

    def validate(self) -> None:
        _require_exact_keys(
            self.raw,
            {
                "schema_version",
                "config_id",
                "weight_groups",
                "layer_exceptions",
                "compute_fidelities",
                "activation_residual_dtype",
                "ccl_dtype",
                "kv_cache_dtype",
                "logits_sampling_dtype_assumptions",
            },
            "precision config",
        )
        if self.raw["schema_version"] != 2:
            raise ValueError(f"unsupported precision schema version {self.raw['schema_version']!r}")
        if not isinstance(self.raw["config_id"], str) or not self.raw["config_id"].strip():
            raise ValueError("config_id must be a non-empty string")
        _require_exact_keys(self.raw["weight_groups"], _WEIGHT_KEYS, "weight_groups")
        _require_exact_keys(self.raw["compute_fidelities"], _FIDELITY_KEYS, "compute_fidelities")
        _require_exact_keys(self.raw["activation_residual_dtype"], _ACTIVATION_KEYS, "activation_residual_dtype")
        _require_exact_keys(self.raw["ccl_dtype"], _CCL_KEYS, "ccl_dtype")
        _require_exact_keys(
            self.raw["logits_sampling_dtype_assumptions"], _LOGITS_KEYS, "logits_sampling_dtype_assumptions"
        )
        _validate_named_values(self.raw["weight_groups"], _DTYPES, "weight_groups")
        _validate_named_values(self.raw["compute_fidelities"], _MATH_FIDELITIES, "compute_fidelities")
        _validate_named_values(self.raw["activation_residual_dtype"], _DTYPES, "activation_residual_dtype")
        _validate_named_values(self.raw["ccl_dtype"], _DTYPES, "ccl_dtype")
        logits = self.raw["logits_sampling_dtype_assumptions"]
        _validate_named_values(
            {
                "lm_head_output": logits["lm_head_output"],
                "topk_values_gather_dtype": logits["topk_values_gather_dtype"],
                "sampling_accumulator": logits["sampling_accumulator"],
            },
            _DTYPES,
            "logits_sampling_dtype_assumptions",
        )
        if not isinstance(logits["full_logits_gather"], dict):
            raise ValueError("full_logits_gather must be an explicit materialization policy object")
        _require_exact_keys(logits["full_logits_gather"], _FULL_LOGITS_GATHER_KEYS, "full_logits_gather")
        gather = logits["full_logits_gather"]
        if gather["mode"] != "not_materialized":
            raise ValueError("the optimized sharded sampler requires full_logits_gather.mode=not_materialized")
        if self.raw["kv_cache_dtype"] not in _DTYPES:
            raise ValueError(f"unsupported kv_cache_dtype {self.raw['kv_cache_dtype']!r}")
        weights = self.raw["weight_groups"]
        if weights["attention_qkv"] != weights["attention_output"]:
            raise ValueError("the packed attention implementation requires matching QKV/output weight dtypes")
        if weights["experts_gate_up"] != weights["experts_down"]:
            raise ValueError("the packed expert implementation requires matching gate/up/down weight dtypes")
        if weights["normalization"] != "bfloat16":
            raise ValueError("the optimized RMSNorm path currently requires bfloat16 normalization weights")
        if self.raw["activation_residual_dtype"]["stack_residual"] != "bfloat16":
            raise ValueError("the optimized residual/norm boundary currently requires bfloat16")
        assumptions = self.raw["logits_sampling_dtype_assumptions"]
        if assumptions["topk_values_gather_dtype"] != "bfloat16":
            raise ValueError("TTSampling's top-k value gather is fixed to bfloat16")
        if assumptions["sampling_accumulator"] != "bfloat16":
            raise ValueError("SamplingGenerator's score accumulator is fixed to bfloat16")
        exceptions = self.raw["layer_exceptions"]
        if not isinstance(exceptions, dict):
            raise ValueError("layer_exceptions must be an object keyed by decimal layer index")
        for layer, override in exceptions.items():
            if not layer.isdecimal() or not 0 <= int(layer) < 36:
                raise ValueError(f"invalid layer exception index {layer!r}")
            merged = _merge_policy(self._base_policy_dict(), override)
            candidate = copy.deepcopy(self.raw)
            candidate.update(merged)
            candidate["layer_exceptions"] = {}
            PrecisionConfig(candidate)

    def terminal_dtypes(self) -> dict[str, object]:
        weights = self.raw["weight_groups"]
        logits = self.raw["logits_sampling_dtype_assumptions"]
        return {
            "embedding": _DTYPES[weights["embedding"]],
            "normalization": _DTYPES[weights["normalization"]],
            "lm_head_weight": _DTYPES[weights["lm_head"]],
            "lm_head_output": _DTYPES[logits["lm_head_output"]],
            "lm_head_math_fidelity": _MATH_FIDELITIES[self.raw["compute_fidelities"]["lm_head"]],
            "full_logits_gather": copy.deepcopy(logits["full_logits_gather"]),
            "topk_values_gather_dtype": _DTYPES[logits["topk_values_gather_dtype"]],
            "sampling_accumulator": _DTYPES[logits["sampling_accumulator"]],
        }

    def to_dict(self) -> dict:
        return copy.deepcopy(self.raw)


def load_precision_config(path: str | Path | None = None) -> PrecisionConfig:
    explicit = path or os.environ.get("GPT_OSS_120B_PRECISION_CONFIG")
    resolved = Path(explicit).expanduser().resolve() if explicit else DEFAULT_PRECISION_CONFIG_PATH
    return PrecisionConfig(json.loads(resolved.read_text(encoding="utf-8")), resolved)


__all__ = [
    "DEFAULT_PRECISION_CONFIG_PATH",
    "PrecisionConfig",
    "dtype_name",
    "load_precision_config",
    "math_fidelity_name",
]
