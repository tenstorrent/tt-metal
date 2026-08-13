# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mechanical precision-policy loading for the Mistral Small 24B autoport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import ttnn

DTYPES = {
    "BF16": ttnn.bfloat16,
    "BFP8_B": ttnn.bfloat8_b,
    "BFP4_B": ttnn.bfloat4_b,
    "UINT32": ttnn.uint32,
}
MATH_FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def dtype_from_name(name: str):
    try:
        return DTYPES[name]
    except KeyError as error:
        raise ValueError(f"unsupported precision-policy dtype {name!r}; expected one of {sorted(DTYPES)}") from error


def fidelity_from_name(name: str):
    try:
        return MATH_FIDELITIES[name]
    except KeyError as error:
        raise ValueError(
            f"unsupported precision-policy math fidelity {name!r}; expected one of {sorted(MATH_FIDELITIES)}"
        ) from error


def dtype_name(dtype) -> str:
    for name, value in DTYPES.items():
        if dtype == value:
            return name
    return str(dtype)


def fidelity_name(fidelity) -> str:
    for name, value in MATH_FIDELITIES.items():
        if fidelity == value:
            return name
    return str(fidelity)


def _group_dtype(policy: Mapping[str, Any], group: str):
    return dtype_from_name(policy["weight_groups"][group]["dtype"])


def _parse_layer_exceptions(policy: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    for layer_text, overrides in policy.get("layer_exceptions", {}).items():
        layer = int(layer_text)
        item: dict[str, Any] = {}
        for key, value in overrides.items():
            if key.endswith("_dtype"):
                item[key] = dtype_from_name(value)
            elif key.endswith("_math_fidelity"):
                item[key] = fidelity_from_name(value)
            else:
                raise ValueError(f"unsupported layer exception field {key!r}")
        parsed[layer] = item
    return parsed


def load_precision_policy(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one checked JSON policy and return raw policy plus FullModelConfig kwargs."""

    policy_path = Path(path).resolve()
    policy = json.loads(policy_path.read_text())
    required = {
        "config_id",
        "weight_groups",
        "compute_fidelities",
        "activation_residual",
        "ccl",
        "kv_cache",
        "logits_sampling",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"precision policy {policy_path} is missing fields: {missing}")

    weights = policy["weight_groups"]
    fidelities = policy["compute_fidelities"]
    activations = policy["activation_residual"]
    ccl = policy["ccl"]
    logits = policy["logits_sampling"]
    runtime = policy.get("runtime", {})
    kwargs = {
        "precision_config_id": str(policy["config_id"]),
        "precision_config_path": str(policy_path),
        "embedding_weight_dtype": _group_dtype(policy, "embedding"),
        "norm_weight_dtype": _group_dtype(policy, "norms"),
        "attention_weight_dtype": _group_dtype(policy, "attention_qkv_wo"),
        "mlp_gate_up_weight_dtype": _group_dtype(policy, "mlp_gate_up"),
        "mlp_down_weight_dtype": _group_dtype(policy, "mlp_down"),
        "lm_head_weight_dtype": _group_dtype(policy, "lm_head"),
        "attention_math_fidelity": fidelity_from_name(fidelities["attention_qkv_wo"]),
        "mlp_math_fidelity": fidelity_from_name(fidelities["mlp_gate_up"]),
        "mlp_down_math_fidelity": fidelity_from_name(fidelities["mlp_down"]),
        "sdpa_math_fidelity": fidelity_from_name(fidelities["sdpa"]),
        "lm_head_math_fidelity": fidelity_from_name(fidelities["lm_head"]),
        "attention_activation_dtype": dtype_from_name(activations["attention_input"]),
        "mlp_activation_dtype": dtype_from_name(activations["mlp_input"]),
        "residual_dtype": dtype_from_name(activations["residual"]),
        "decode_collective_dtype": dtype_from_name(ccl["decode_payload"]),
        "prefill_collective_dtype": dtype_from_name(ccl["prefill_payload"]),
        "collective_workspace_dtype": dtype_from_name(ccl["workspace"]),
        "kv_cache_dtype": dtype_from_name(policy["kv_cache"]["dtype"]),
        "logits_dtype": dtype_from_name(logits["logits"]),
        "sampling_logits_dtype": dtype_from_name(logits["sampling_logits"]),
        "sampling_params_dtype": dtype_from_name(logits["sampling_params"]),
        "sampling_index_dtype": dtype_from_name(logits["token_indices"]),
        "layer_exceptions": _parse_layer_exceptions(policy),
    }
    if "max_context_len" in runtime:
        kwargs["max_context_len"] = int(runtime["max_context_len"])
    if "attention_geometry" in runtime:
        kwargs["attention_geometry"] = tuple(int(value) for value in runtime["attention_geometry"])
    if "mlp_geometry" in runtime:
        kwargs["mlp_geometry"] = tuple(int(value) for value in runtime["mlp_geometry"])
    return policy, kwargs


__all__ = [
    "dtype_from_name",
    "dtype_name",
    "fidelity_from_name",
    "fidelity_name",
    "load_precision_policy",
]
