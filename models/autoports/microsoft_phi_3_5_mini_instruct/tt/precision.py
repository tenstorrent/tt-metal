# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Precision policy loading for the Phi-3.5-mini TTNN autoport."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ttnn


MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_CONFIG = MODEL_DIR / "doc" / "datatype_sweep" / "selected_precision_config.json"


_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bf16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bf8": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "bf4": ttnn.bfloat4_b,
    "float32": ttnn.float32,
    "fp32": ttnn.float32,
}


DEFAULT_PRECISION_CONFIG: dict[str, Any] = {
    "config_id": "default_optimized",
    "description": "Optimized full-model default prior to datatype sweep selection.",
    "weight_groups": {
        "embedding": {"dtype": "bfloat16"},
        "attention.qkv": {"decode_dtype": "bfloat8_b", "prefill_dtype": "bfloat8_b"},
        "attention.o": {"decode_dtype": "bfloat8_b", "prefill_dtype": "bfloat8_b"},
        "mlp.gate_up": {"decode_dtype": "bfloat4_b", "prefill_dtype": "bfloat8_b"},
        "mlp.down": {"decode_dtype": "bfloat4_b", "prefill_dtype": "bfloat8_b"},
        "norm": {"dtype": "bfloat16"},
        "lm_head": {"decode_dtype": "bfloat16", "prefill_dtype": "bfloat16"},
    },
    "layer_exceptions": [],
    "activation_dtype": "bfloat16",
    "residual_dtype": "bfloat16",
    "ccl": {
        "policy": "sync_all_reduce",
        "decode_dtype": "bfloat8_b",
        "prefill_dtype": "bfloat16",
    },
    "kv_cache_dtype": "bfloat8_b",
    "compute_fidelity": {
        "decode_matmul": "lofi",
        "prefill_matmul": "hifi2",
        "norm": "hifi4",
        "lm_head_decode": "lofi",
        "lm_head_prefill": "hifi2",
        "sdpa_decode": "lofi",
        "sdpa_prefill": "hifi4",
    },
    "logits_dtype": "bfloat16",
    "sampling_dtype": "bfloat16 logits into canonical split sampler; sampled token is uint32",
}


@dataclass(frozen=True)
class Phi35MiniPrecisionPolicy:
    raw: dict[str, Any]
    source_path: Path | None = None

    @property
    def config_id(self) -> str:
        return str(self.raw.get("config_id", "unnamed_precision_config"))

    @property
    def activation_dtype(self) -> ttnn.DataType:
        return dtype_from_name(str(self.raw.get("activation_dtype", "bfloat16")))

    @property
    def residual_dtype(self) -> ttnn.DataType:
        return dtype_from_name(str(self.raw.get("residual_dtype", self.raw.get("activation_dtype", "bfloat16"))))

    @property
    def kv_cache_dtype(self) -> ttnn.DataType:
        return dtype_from_name(str(self.raw.get("kv_cache_dtype", "bfloat8_b")))

    @property
    def embedding_dtype(self) -> ttnn.DataType:
        group = self.weight_group("embedding")
        return dtype_from_name(str(group.get("dtype", "bfloat16")))

    @property
    def norm_dtype(self) -> ttnn.DataType:
        group = self.weight_group("norm")
        return dtype_from_name(str(group.get("dtype", "bfloat16")))

    @property
    def logits_dtype(self) -> ttnn.DataType:
        return dtype_from_name(str(self.raw.get("logits_dtype", "bfloat16")))

    @property
    def ccl_policy(self) -> str:
        return str(self.raw.get("ccl", {}).get("policy", "sync_all_reduce"))

    @property
    def decode_ccl_dtype(self) -> ttnn.DataType:
        return dtype_from_name(str(self.raw.get("ccl", {}).get("decode_dtype", "bfloat8_b")))

    @property
    def prefill_ccl_dtype(self) -> ttnn.DataType:
        return dtype_from_name(str(self.raw.get("ccl", {}).get("prefill_dtype", "bfloat16")))

    def compute_fidelity(self, key: str, default: str) -> str:
        return str(self.raw.get("compute_fidelity", {}).get(key, default))

    def weight_group(self, name: str, *, layer_idx: int | None = None) -> dict[str, Any]:
        group = dict((self.raw.get("weight_groups") or {}).get(name, {}))
        if layer_idx is None:
            return group
        for exception in self.raw.get("layer_exceptions", []) or []:
            layers = exception.get("layers", [])
            if _layer_matches(layer_idx, layers):
                override = (exception.get("weight_groups") or {}).get(name)
                if override:
                    group.update(override)
        return group

    def weight_dtype(self, name: str, *, layer_idx: int | None = None, prefill: bool = False) -> ttnn.DataType:
        group = self.weight_group(name, layer_idx=layer_idx)
        key = "prefill_dtype" if prefill else "decode_dtype"
        dtype_name = group.get(key, group.get("dtype", "bfloat16"))
        return dtype_from_name(str(dtype_name))

    def to_summary(self) -> dict[str, Any]:
        summary = dict(self.raw)
        summary["source_path"] = str(self.source_path) if self.source_path is not None else "built_in_default"
        return summary


def load_precision_policy(
    model_dir: str | Path | None = None,
    precision_config_path: str | Path | None = None,
) -> Phi35MiniPrecisionPolicy:
    path = _resolve_precision_config_path(model_dir=model_dir, precision_config_path=precision_config_path)
    if path is None:
        return Phi35MiniPrecisionPolicy(raw=json.loads(json.dumps(DEFAULT_PRECISION_CONFIG)), source_path=None)
    data = json.loads(path.read_text())
    return Phi35MiniPrecisionPolicy(raw=_merge_defaults(data), source_path=path)


def dtype_from_name(name: str) -> ttnn.DataType:
    normalized = name.lower()
    if normalized not in _DTYPES:
        raise ValueError(f"unsupported Phi-3.5 precision dtype {name!r}")
    return _DTYPES[normalized]


def dtype_name(dtype: ttnn.DataType) -> str:
    if dtype == ttnn.bfloat16:
        return "bfloat16"
    if dtype == ttnn.bfloat8_b:
        return "bfloat8_b"
    if dtype == ttnn.bfloat4_b:
        return "bfloat4_b"
    if dtype == ttnn.float32:
        return "float32"
    return str(dtype)


def _resolve_precision_config_path(
    *,
    model_dir: str | Path | None,
    precision_config_path: str | Path | None,
) -> Path | None:
    explicit = precision_config_path or os.getenv("PHI35_PRECISION_CONFIG")
    if explicit:
        if str(explicit).lower() in {"default", "builtin", "none"}:
            return None
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"PHI35_PRECISION_CONFIG does not exist: {path}")
        return path

    root = Path(model_dir).resolve() if model_dir is not None else MODEL_DIR
    selected = root / "doc" / "datatype_sweep" / "selected_precision_config.json"
    if selected.exists():
        return selected
    if DEFAULT_SELECTED_CONFIG.exists():
        return DEFAULT_SELECTED_CONFIG
    return None


def _merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_PRECISION_CONFIG))
    _deep_update(merged, data)
    return merged


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


def _layer_matches(layer_idx: int, layers: Any) -> bool:
    if layers == "all":
        return True
    if isinstance(layers, int):
        return layer_idx == layers
    if isinstance(layers, str):
        if "-" in layers:
            start, end = layers.split("-", 1)
            return int(start) <= layer_idx <= int(end)
        return layer_idx == int(layers)
    if isinstance(layers, list):
        return any(_layer_matches(layer_idx, item) for item in layers)
    return False
