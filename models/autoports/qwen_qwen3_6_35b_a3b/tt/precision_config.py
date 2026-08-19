# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision policy loading for the Qwen3.6-35B-A3B autoport."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import ttnn

DEFAULT_PRECISION_CONFIG: dict[str, Any] = {
    "config_id": "baseline_default",
    "description": "Optimized full-model baseline precision policy.",
    "weight_groups": {
        "embedding": {"dtype": "bf16"},
        "norms": {"dtype": "bf16"},
        "attention": {"dtype": "bf8"},
        "linear_attention": {"dtype": "bf8"},
        "router": {"dtype": "bf16"},
        "shared_moe": {"dtype": "bf8"},
        "routed_moe": {"dtype": {"linear_attention": "bf8", "full_attention": "bf4"}},
        "lm_head": {"dtype": "bf8"},
    },
    "layer_exceptions": {},
    "compute_fidelities": {
        "attention": "default",
        "linear_attention": "default",
        "router": "default",
        "shared_moe": "default",
        "routed_moe": "default",
        "lm_head": "default",
    },
    "activation_dtype": "bf16",
    "residual_dtype": "bf16",
    "ccl_dtype": "bf16",
    "kv_cache_dtype": "bf16",
    "linear_state_dtype": "bf16",
    "logits_dtype": "bf16",
    "sampling_dtype": "uint32",
    "max_top_k": 32,
}


_DTYPE_NAMES = {
    "bf16": ttnn.bfloat16,
    "bfloat16": ttnn.bfloat16,
    "bf8": ttnn.bfloat8_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfp8": ttnn.bfloat8_b,
    "bf4": ttnn.bfloat4_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "bfp4": ttnn.bfloat4_b,
    "uint32": ttnn.uint32,
    "int32": ttnn.int32,
}


_MATH_FIDELITY_NAMES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def selected_precision_config_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "doc" / "datatype_sweep" / "selected_precision_config.json"


def dtype_from_name(name: str):
    key = str(name).strip().lower()
    try:
        return _DTYPE_NAMES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen precision dtype {name!r}") from exc


def dtype_to_name(dtype) -> str:
    for name, value in _DTYPE_NAMES.items():
        if dtype == value and name in {"bf16", "bf8", "bf4", "uint32", "int32"}:
            return name
    return str(dtype)


def math_fidelity_from_name(name: str | None):
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in {"", "none", "null", "default", "ttnn_default"}:
        return None
    try:
        return _MATH_FIDELITY_NAMES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen compute fidelity {name!r}") from exc


def compute_kernel_config_from_fidelity(name: str | None):
    fidelity = math_fidelity_from_name(name)
    if fidelity is None:
        return None
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def normalize_precision_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return _deep_update(DEFAULT_PRECISION_CONFIG, config or {})


def load_precision_config(
    *,
    model_dir: str | Path,
    precision_config: str | Path | dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Load explicit/env/default selected precision config for model construction."""

    if isinstance(precision_config, dict):
        return normalize_precision_config(precision_config), "<in-memory>"

    path: Path | None = None
    if precision_config is not None:
        path = Path(precision_config)
    else:
        env_path = os.environ.get("QWEN36_PRECISION_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            default_path = selected_precision_config_path(model_dir)
            if default_path.exists():
                path = default_path

    if path is None:
        return normalize_precision_config(None), "<built-in-default>"

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    config = normalize_precision_config(data)
    config["source_path"] = str(path)
    return config, str(path)


def group_dtype_name(config: dict[str, Any], group: str, *, layer_type: str | None = None) -> str:
    value = config["weight_groups"][group]["dtype"]
    if isinstance(value, dict):
        if layer_type is None:
            raise ValueError(f"group {group!r} has layer-type dtype choices but no layer_type was supplied")
        value = value[layer_type]
    return str(value)


def compute_fidelity_name(config: dict[str, Any], group: str) -> str:
    return str(config.get("compute_fidelities", {}).get(group, "default"))


def layer_exception(config: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    exceptions = config.get("layer_exceptions") or {}
    return copy.deepcopy(exceptions.get(str(layer_idx), exceptions.get(layer_idx, {})))
