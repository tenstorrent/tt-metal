# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resolve weight and cache precision for the standalone Galaxy prefill model."""

import json
import os

import ttnn

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precision_overrides.json")

# Module names that may be overridden — keep in sync with the JSON schema and
# with the constructors that accept these kwargs (Gemma4Model and below).
KNOWN_MODULES = ("shared_mlp", "attention", "lm_head", "embedding", "kv_cache")

_DTYPE_BY_NAME = {
    "bf16": ttnn.bfloat16,
    "bfloat16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "fp32": ttnn.float32,
    "float32": ttnn.float32,
}


def dtype_to_str(dtype):
    """Short stable string for cache-filename suffixes ("bf16" / "bfp8" / "bfp4" / "fp32").

    Cache filenames embed the dtype string so flipping a module's dtype in
    precision_overrides.json doesn't reuse a stale cached tensor at the
    previous precision.
    """
    if dtype == ttnn.bfloat16:
        return "bf16"
    if dtype == ttnn.bfloat8_b:
        return "bfp8"
    if dtype == ttnn.bfloat4_b:
        return "bfp4"
    if dtype == ttnn.float32:
        return "fp32"
    raise ValueError(f"No cache-suffix mapping for dtype {dtype}")


class Gemma4Precision:
    """Per-module dtype mapping. Construct via ``Gemma4Precision.load(...)``
    or directly with ``Gemma4Precision({...})``."""

    def __init__(self, overrides=None):
        self._overrides = dict(overrides) if overrides else {}

    def get(self, module_name, default=ttnn.bfloat16):
        return self._overrides.get(module_name, default)

    def __repr__(self):
        return f"Gemma4Precision({self._overrides!r})"

    @classmethod
    def load(cls, model_path):
        """Resolve module precision overrides using the model path basename."""
        try:
            with open(_PATH) as f:
                table = json.load(f)
        except OSError as exc:
            raise RuntimeError(f"Cannot read required precision configuration {_PATH}: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSON in precision configuration {_PATH}: {exc}") from exc

        model_key = os.path.basename(str(model_path).rstrip("/"))
        model_entry = table.get(model_key)
        if not model_entry:
            raise ValueError(
                f"No precision configuration for {model_path!r} (model key {model_key!r}) in {_PATH}; "
                f"expected one of {sorted(table)}"
            )

        resolved = {}
        for k, v in model_entry.items():
            if k not in KNOWN_MODULES:
                continue  # ignore unknown / future keys silently
            if v not in _DTYPE_BY_NAME:
                raise ValueError(
                    f"precision_overrides.json[{model_key}][{k}]={v!r} — "
                    f"unknown dtype; expected one of {sorted(_DTYPE_BY_NAME)}"
                )
            resolved[k] = _DTYPE_BY_NAME[v]
        return cls(resolved)
