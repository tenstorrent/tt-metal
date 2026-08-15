# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mechanical precision-policy loading for the Gemma-4 autoport."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import ttnn

DEFAULT_POLICY_PATH = Path(__file__).parents[1] / "doc" / "datatype_sweep" / "selected_precision_config.json"

_DTYPES = {
    "BF16": ttnn.bfloat16,
    "BFP8_B": ttnn.bfloat8_b,
    "BFP4_B": ttnn.bfloat4_b,
    "FP32": ttnn.float32,
}
_FIDELITIES = {
    "LOFI": ttnn.MathFidelity.LoFi,
    "HIFI2": ttnn.MathFidelity.HiFi2,
    "HIFI4": ttnn.MathFidelity.HiFi4,
}


def load_precision_policy(path: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    configured = path or os.getenv("GEMMA4_PRECISION_CONFIG")
    resolved = Path(configured) if configured else DEFAULT_POLICY_PATH
    if not resolved.exists():
        return {}, None
    with resolved.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    if "extends" in policy:
        base, _ = load_precision_policy(resolved.parent / policy["extends"])
        policy = _deep_merge(base, policy.get("overrides", {}))
        policy["config_id"] = json.loads(resolved.read_text(encoding="utf-8"))["config_id"]
    return policy, resolved.resolve()


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _dtype(name: str) -> Any:
    try:
        return _DTYPES[name.upper()]
    except KeyError as error:
        raise ValueError(f"unsupported Gemma-4 policy dtype {name!r}") from error


def _fidelity(name: str) -> Any:
    try:
        return _FIDELITIES[name.upper()]
    except KeyError as error:
        raise ValueError(f"unsupported Gemma-4 policy fidelity {name!r}") from error


def layer_policy(policy: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    resolved = deepcopy(policy)
    for exception in policy.get("layer_exceptions", []):
        if layer_idx in exception.get("layers", []):
            for section, values in exception.get("overrides", {}).items():
                resolved.setdefault(section, {}).update(values)
    return resolved


def decoder_kwargs(policy: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    selected = layer_policy(policy, layer_idx)
    weights = selected.get("weight_groups", {})
    fidelity = selected.get("compute_fidelities", {})
    result: dict[str, Any] = {}
    weight_mapping = {
        "attention_qkv_o": "attention_weight_dtype",
        "dense_gate_up": "mlp_weight_dtype",
        "dense_down": "mlp_down_weight_dtype",
        "experts_gate_up_down": "expert_weight_dtype",
        "prefill_experts": "prefill_expert_weight_dtype",
        "norms": "weight_dtype",
        "router_and_routing": "router_weight_dtype",
    }
    fidelity_mapping = {
        "sliding_attention": "attention_math_fidelity",
        "full_attention": "full_attention_math_fidelity",
        "dense_mlp": "mlp_math_fidelity",
        "expert_gate_up": "expert_gate_math_fidelity",
        "expert_down": "expert_math_fidelity",
    }
    for field, kwarg in weight_mapping.items():
        if field in weights:
            result[kwarg] = _dtype(weights[field])
    for field, kwarg in fidelity_mapping.items():
        if field in fidelity:
            result[kwarg] = _fidelity(fidelity[field])
    activation = selected.get("activation_residual", {})
    if "activation_dtype" in activation:
        result["activation_dtype"] = _dtype(activation["activation_dtype"])
    if selected.get("decode_weight_groups"):
        result["decode_weight_dtypes"] = {
            role: _dtype(dtype) for role, dtype in selected["decode_weight_groups"].items()
        }
    return result


def dtype_from_policy(policy: dict[str, Any], section: str, field: str, default: Any) -> Any:
    value = policy.get(section, {}).get(field)
    return default if value is None else _dtype(value)


def weight_dtype_from_policy(policy: dict[str, Any], field: str, default: Any) -> Any:
    value = policy.get("weight_groups", {}).get(field)
    return default if value is None else _dtype(value)


def dtype_name(value: Any) -> str:
    return next((name for name, dtype in _DTYPES.items() if dtype == value), str(value))


def fidelity_name(value: Any) -> str:
    return next((name for name, fidelity in _FIDELITIES.items() if fidelity == value), str(value))
