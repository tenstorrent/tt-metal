# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mechanically consumed precision policy for the Qwen3.6-27B full model."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizationPolicy, resolve_policy

MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECISION_CONFIG = MODEL_ROOT / "doc/datatype_sweep/selected_precision_config.json"
PRECISION_CONFIG_ENV = "QWEN36_PRECISION_CONFIG"

DTYPES = {
    "BF16": ttnn.bfloat16,
    "BFP8_B": ttnn.bfloat8_b,
    "BFP4_B": ttnn.bfloat4_b,
    "FP32": ttnn.float32,
    "UINT32": ttnn.uint32,
}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def _require_keys(value: Mapping, expected: set[str], where: str) -> None:
    missing, extra = expected - set(value), set(value) - expected
    if missing or extra:
        raise ValueError(f"{where} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def _dtype(name: str):
    try:
        return DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(DTYPES)}") from exc


def _fidelity(name: str):
    try:
        return FIDELITIES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported fidelity {name!r}; expected one of {sorted(FIDELITIES)}") from exc


@dataclass(frozen=True)
class PrecisionConfig:
    source: str
    raw: Mapping
    sha256: str

    @property
    def config_id(self) -> str:
        return str(self.raw["config_id"])

    @property
    def activation_residual_dtype(self):
        return _dtype(self.raw["activation_residual_dtype"])

    @property
    def lm_head_weight_dtype(self):
        return _dtype(self.raw["logits_sampling"]["lm_head_weight_dtype"])

    @property
    def lm_head_output_dtype(self):
        return _dtype(self.raw["logits_sampling"]["lm_head_output_dtype"])

    @property
    def lm_head_fidelity(self):
        return _fidelity(self.raw["logits_sampling"]["lm_head_compute_fidelity"])

    def ccl_dtype(self, role: str):
        return _dtype(self.raw["ccl_dtype"][role])

    def policy_for(self, layer_idx: int, layer_kind: str) -> OptimizationPolicy:
        policy = resolve_policy(self.raw["base_policy"][layer_kind], layer_kind)
        fields = self._policy_fields(layer_kind)
        for exception in self.raw["layer_exceptions"]:
            if layer_idx in exception["layers"] and exception["layer_kind"] in ("any", layer_kind):
                fields.update(exception["overrides"])
        return replace(policy, **fields)

    def _policy_fields(self, layer_kind: str) -> dict:
        weights = self.raw["weight_groups"]
        fidelity = self.raw["compute_fidelities"]
        if layer_kind == "full_attention":
            return {
                "activation_residual_dtype": self.activation_residual_dtype,
                "attention_weight_dtype": _dtype(weights["full_attention_qkv"]),
                "mlp_gate_up_dtype": _dtype(weights["mlp_gate_up"]),
                "mlp_down_dtype": _dtype(weights["mlp_down"]),
                "cache_dtype": _dtype(self.raw["kv_cache_dtype"]),
                "attention_fidelity": _fidelity(fidelity["full_attention_sdpa"]),
                "qkv_fidelity": _fidelity(fidelity["full_attention_qkv"]),
                "o_fidelity": _fidelity(fidelity["full_attention_o"]),
                "mlp_fidelity": _fidelity(fidelity["mlp"]),
            }
        return {
            "activation_residual_dtype": self.activation_residual_dtype,
            "attention_weight_dtype": _dtype(weights["linear_attention_internal"]),
            "mlp_gate_up_dtype": _dtype(weights["mlp_gate_up"]),
            "mlp_down_dtype": _dtype(weights["mlp_down"]),
            "cache_dtype": _dtype(self.raw["kv_cache_dtype"]),
            "attention_fidelity": _fidelity(fidelity["linear_attention_internal"]),
            "mlp_fidelity": _fidelity(fidelity["mlp"]),
            "linear_input_weight_dtype": _dtype(weights["linear_input_projection"]),
            "linear_output_weight_dtype": _dtype(weights["linear_output_projection"]),
            "linear_input_fidelity": _fidelity(fidelity["linear_input_projection"]),
            "linear_output_fidelity": _fidelity(fidelity["linear_output_projection"]),
            "linear_recurrent_fidelity": _fidelity(fidelity["linear_recurrent"]),
            "linear_recurrent_state_dtype": _dtype(self.raw["linear_recurrent_state_dtype"]),
        }

    def summary(self) -> dict:
        return {"source": self.source, "sha256": self.sha256, **self.raw}


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "config_id",
    "base_policy",
    "weight_groups",
    "layer_exceptions",
    "compute_fidelities",
    "activation_residual_dtype",
    "ccl_dtype",
    "kv_cache_dtype",
    "linear_recurrent_state_dtype",
    "logits_sampling",
}


def _validate(raw: Mapping) -> None:
    _require_keys(raw, REQUIRED_TOP_LEVEL, "precision config")
    if raw["schema_version"] != 1:
        raise ValueError("precision config schema_version must be 1")
    _require_keys(raw["base_policy"], {"full_attention", "linear_attention"}, "base_policy")
    _require_keys(
        raw["weight_groups"],
        {
            "full_attention_qkv",
            "full_attention_o",
            "linear_attention_internal",
            "linear_input_projection",
            "linear_output_projection",
            "mlp_gate_up",
            "mlp_down",
        },
        "weight_groups",
    )
    # The optimized decoder uses a common storage tensor for full-attention
    # QKV and O. Reject an unrepresentable split rather than silently ignoring it.
    if raw["weight_groups"]["full_attention_qkv"] != raw["weight_groups"]["full_attention_o"]:
        raise ValueError("full_attention_qkv and full_attention_o dtypes must match in the current packed runtime")
    _require_keys(
        raw["compute_fidelities"],
        {
            "full_attention_sdpa",
            "full_attention_qkv",
            "full_attention_o",
            "linear_attention_internal",
            "linear_input_projection",
            "linear_output_projection",
            "linear_recurrent",
            "mlp",
        },
        "compute_fidelities",
    )
    _require_keys(raw["ccl_dtype"], {"token_mixer", "mlp"}, "ccl_dtype")
    _require_keys(
        raw["logits_sampling"],
        {
            "lm_head_weight_dtype",
            "lm_head_output_dtype",
            "lm_head_compute_fidelity",
            "sampling_logits_dtype",
            "sampled_token_dtype",
        },
        "logits_sampling",
    )
    if raw["logits_sampling"]["sampling_logits_dtype"] != raw["logits_sampling"]["lm_head_output_dtype"]:
        raise ValueError("sampling_logits_dtype must match lm_head_output_dtype; there is no intervening cast")
    if raw["logits_sampling"]["sampled_token_dtype"] != "UINT32":
        raise ValueError("the common sampler requires UINT32 sampled tokens")
    for name in raw["weight_groups"].values():
        _dtype(name)
    for name in raw["compute_fidelities"].values():
        _fidelity(name)
    _dtype(raw["activation_residual_dtype"])
    _dtype(raw["kv_cache_dtype"])
    _dtype(raw["linear_recurrent_state_dtype"])
    for name in raw["ccl_dtype"].values():
        if name not in ("BF16", "BFP8_B"):
            raise ValueError("CCL dtype must be BF16 or BFP8_B")
    for exception in raw["layer_exceptions"]:
        _require_keys(exception, {"layers", "layer_kind", "overrides"}, "layer exception")
        if exception["layer_kind"] not in ("any", "full_attention", "linear_attention"):
            raise ValueError("invalid layer exception kind")
        if not all(isinstance(index, int) and index >= 0 for index in exception["layers"]):
            raise ValueError("layer exception indices must be non-negative integers")
        unknown = set(exception["overrides"]) - set(OptimizationPolicy.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown OptimizationPolicy overrides: {sorted(unknown)}")


def load_precision_config(value=None) -> PrecisionConfig:
    if isinstance(value, PrecisionConfig):
        return value
    source = "built-in-safe-baseline"
    if value is None:
        value = os.environ.get(PRECISION_CONFIG_ENV)
        if value is None and DEFAULT_PRECISION_CONFIG.exists():
            value = DEFAULT_PRECISION_CONFIG
    if value is None:
        raw = safe_baseline_config()
    elif isinstance(value, Mapping):
        raw, source = dict(value), "mapping"
    else:
        path = Path(value)
        raw, source = json.loads(path.read_text()), str(path.resolve())
    _validate(raw)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return PrecisionConfig(source=source, raw=raw, sha256=hashlib.sha256(canonical).hexdigest())


def safe_baseline_config() -> dict:
    return {
        "schema_version": 1,
        "config_id": "baseline_optimized_default",
        "base_policy": {"full_attention": "final_cumulative", "linear_attention": "linear_final"},
        "weight_groups": {
            "full_attention_qkv": "BF16",
            "full_attention_o": "BF16",
            "linear_attention_internal": "BFP4_B",
            "linear_input_projection": "BFP4_B",
            "linear_output_projection": "BFP4_B",
            "mlp_gate_up": "BFP4_B",
            "mlp_down": "BFP4_B",
        },
        "layer_exceptions": [],
        "compute_fidelities": {
            "full_attention_sdpa": "HiFi4",
            "full_attention_qkv": "HiFi2",
            "full_attention_o": "HiFi2",
            "linear_attention_internal": "LoFi",
            "linear_input_projection": "LoFi",
            "linear_output_projection": "LoFi",
            "linear_recurrent": "HiFi2",
            "mlp": "LoFi",
        },
        "activation_residual_dtype": "BF16",
        "ccl_dtype": {"token_mixer": "BF16", "mlp": "BF16"},
        "kv_cache_dtype": "BFP8_B",
        "linear_recurrent_state_dtype": "BFP8_B",
        "logits_sampling": {
            "lm_head_weight_dtype": "BFP8_B",
            "lm_head_output_dtype": "BFP8_B",
            "lm_head_compute_fidelity": "HiFi2",
            "sampling_logits_dtype": "BFP8_B",
            "sampled_token_dtype": "UINT32",
        },
    }


__all__ = ["DEFAULT_PRECISION_CONFIG", "PRECISION_CONFIG_ENV", "PrecisionConfig", "load_precision_config"]
