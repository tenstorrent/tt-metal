# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mechanically consumed precision policies for Gemma 4 31B Stage 08."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import ttnn
from models.autoports.google_gemma_4_31b.tt.multichip_decoder import DEFAULT_MULTICHIP_OPTIMIZATION_POLICY
from models.autoports.google_gemma_4_31b.tt.optimized_decoder import DecoderOptimizationPolicy

DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "float32": ttnn.float32,
}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
MATERIAL_WEIGHT_GROUPS = (
    "attention_prefill",
    "attention_qkv",
    "attention_output",
    "mlp_gate_up",
    "mlp_down",
    "lm_head",
)


def dtype_name(dtype: ttnn.DataType) -> str:
    for name, value in DTYPES.items():
        if dtype == value:
            return name
    raise ValueError(f"unsupported Gemma 4 precision dtype: {dtype}")


def fidelity_name(fidelity: ttnn.MathFidelity) -> str:
    for name, value in FIDELITIES.items():
        if fidelity == value:
            return name
    raise ValueError(f"unsupported Gemma 4 math fidelity: {fidelity}")


def _dtype(value: str, field: str) -> ttnn.DataType:
    try:
        return DTYPES[value]
    except KeyError as error:
        raise ValueError(f"{field} has unsupported dtype {value!r}; expected one of {sorted(DTYPES)}") from error


def _fidelity(value: str, field: str) -> ttnn.MathFidelity:
    try:
        return FIDELITIES[value]
    except KeyError as error:
        raise ValueError(
            f"{field} has unsupported math fidelity {value!r}; expected one of {sorted(FIDELITIES)}"
        ) from error


def _group_dtype(data: dict[str, Any], group: str) -> ttnn.DataType:
    try:
        value = data["weight_groups"][group]["dtype"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"precision config is missing weight_groups.{group}.dtype") from error
    return _dtype(value, f"weight_groups.{group}.dtype")


def _group_fidelity(data: dict[str, Any], group: str) -> ttnn.MathFidelity:
    try:
        value = data["compute_fidelities"][group]
    except (KeyError, TypeError) as error:
        raise ValueError(f"precision config is missing compute_fidelities.{group}") from error
    return _fidelity(value, f"compute_fidelities.{group}")


def _decoder_policy(data: dict[str, Any], *, name: str) -> DecoderOptimizationPolicy:
    return replace(
        DEFAULT_MULTICHIP_OPTIMIZATION_POLICY,
        name=name,
        attention_weight_dtype=_group_dtype(data, "attention_prefill"),
        attention_qkv_weight_dtype=_group_dtype(data, "attention_qkv"),
        attention_o_weight_dtype=_group_dtype(data, "attention_output"),
        attention_qkv_math_fidelity=_group_fidelity(data, "attention_qkv"),
        attention_o_math_fidelity=_group_fidelity(data, "attention_output"),
        mlp_gate_up_weight_dtype=_group_dtype(data, "mlp_gate_up"),
        mlp_down_weight_dtype=_group_dtype(data, "mlp_down"),
        mlp_gate_up_math_fidelity=_group_fidelity(data, "mlp_gate_up"),
        mlp_down_math_fidelity=_group_fidelity(data, "mlp_down"),
        kv_cache_dtype=_dtype(data["kv_cache_dtype"], "kv_cache_dtype"),
    )


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_data(source: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = set() if seen is None else seen
    if source in seen:
        raise ValueError(f"precision config inheritance cycle at {source}")
    seen.add(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    base_name = data.pop("extends", None)
    if base_name is None:
        return data
    base_path = (source.parent / base_name).resolve()
    return _deep_merge(_load_data(base_path, seen), data)


@dataclass(frozen=True)
class ResolvedPrecisionConfig:
    config_id: str
    source_path: Path
    raw: dict[str, Any]
    default_decoder_policy: DecoderOptimizationPolicy
    layer_decoder_policies: tuple[tuple[int, DecoderOptimizationPolicy], ...]
    activation_dtype: ttnn.DataType
    residual_dtype: ttnn.DataType
    prefill_ccl_dtype: ttnn.DataType
    decode_ccl_dtype: ttnn.DataType
    lm_head_weight_dtype: ttnn.DataType
    lm_head_math_fidelity: ttnn.MathFidelity
    logits_dtype: ttnn.DataType
    sampling_dtype: ttnn.DataType

    def policy_for_layer(self, layer_idx: int) -> DecoderOptimizationPolicy:
        return dict(self.layer_decoder_policies).get(layer_idx, self.default_decoder_policy)

    def summary(self) -> dict[str, Any]:
        policy = self.default_decoder_policy
        return {
            "config_id": self.config_id,
            "source_path": str(self.source_path),
            "weight_groups": {
                "attention_prefill": dtype_name(policy.attention_weight_dtype),
                "attention_qkv": dtype_name(policy.resolved_attention_qkv_weight_dtype),
                "attention_output": dtype_name(policy.resolved_attention_o_weight_dtype),
                "mlp_gate_up": dtype_name(policy.mlp_gate_up_weight_dtype),
                "mlp_down": dtype_name(policy.mlp_down_weight_dtype),
                "lm_head": dtype_name(self.lm_head_weight_dtype),
            },
            "compute_fidelities": {
                "attention_qkv": fidelity_name(policy.resolved_attention_qkv_math_fidelity),
                "attention_output": fidelity_name(policy.resolved_attention_o_math_fidelity),
                "mlp_gate_up": fidelity_name(policy.mlp_gate_up_math_fidelity),
                "mlp_down": fidelity_name(policy.mlp_down_math_fidelity),
                "lm_head": fidelity_name(self.lm_head_math_fidelity),
            },
            "activation_dtype": dtype_name(self.activation_dtype),
            "residual_dtype": dtype_name(self.residual_dtype),
            "ccl_dtype": {
                "prefill": dtype_name(self.prefill_ccl_dtype),
                "decode": dtype_name(self.decode_ccl_dtype),
            },
            "kv_cache_dtype": dtype_name(policy.kv_cache_dtype),
            "logits_dtype": dtype_name(self.logits_dtype),
            "sampling_dtype": dtype_name(self.sampling_dtype),
            "layer_exceptions": [layer for layer, _ in self.layer_decoder_policies],
        }


def load_precision_config(path: str | Path) -> ResolvedPrecisionConfig:
    source = Path(path).expanduser().resolve()
    data = _load_data(source)
    if data.get("schema_version") != 1:
        raise ValueError("precision config schema_version must be 1")
    if data.get("model") != "google/gemma-4-31B":
        raise ValueError("precision config model must be google/gemma-4-31B")
    config_id = data.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise ValueError("precision config requires a non-empty config_id")
    missing_groups = [group for group in MATERIAL_WEIGHT_GROUPS if group not in data.get("weight_groups", {})]
    if missing_groups:
        raise ValueError(f"precision config is missing material weight groups: {missing_groups}")

    default_policy = _decoder_policy(data, name=config_id)
    exceptions = []
    seen_layers: set[int] = set()
    for item in data.get("layer_exceptions", []):
        if not isinstance(item, dict) or not isinstance(item.get("layers"), list):
            raise ValueError("each layer exception requires a layers list")
        merged = json.loads(json.dumps(data))
        for group, override in item.get("weight_groups", {}).items():
            if group not in MATERIAL_WEIGHT_GROUPS or group == "lm_head":
                raise ValueError(f"unsupported per-layer weight group override: {group}")
            merged["weight_groups"][group].update(override)
        merged["compute_fidelities"].update(item.get("compute_fidelities", {}))
        for layer in item["layers"]:
            layer = int(layer)
            if not 0 <= layer < 60 or layer in seen_layers:
                raise ValueError(f"invalid or duplicate layer exception: {layer}")
            seen_layers.add(layer)
            exceptions.append((layer, _decoder_policy(merged, name=f"{config_id}_layer_{layer}")))

    sampling = data.get("sampling", {})
    return ResolvedPrecisionConfig(
        config_id=config_id,
        source_path=source,
        raw=data,
        default_decoder_policy=default_policy,
        layer_decoder_policies=tuple(exceptions),
        activation_dtype=_dtype(data["activation_dtype"], "activation_dtype"),
        residual_dtype=_dtype(data["residual_dtype"], "residual_dtype"),
        prefill_ccl_dtype=_dtype(data["ccl_dtype"]["prefill"], "ccl_dtype.prefill"),
        decode_ccl_dtype=_dtype(data["ccl_dtype"]["decode"], "ccl_dtype.decode"),
        lm_head_weight_dtype=_group_dtype(data, "lm_head"),
        lm_head_math_fidelity=_group_fidelity(data, "lm_head"),
        logits_dtype=_dtype(data["logits_dtype"], "logits_dtype"),
        sampling_dtype=_dtype(sampling["gather_values_dtype"], "sampling.gather_values_dtype"),
    )


__all__ = [
    "DTYPES",
    "FIDELITIES",
    "MATERIAL_WEIGHT_GROUPS",
    "ResolvedPrecisionConfig",
    "dtype_name",
    "fidelity_name",
    "load_precision_config",
]
