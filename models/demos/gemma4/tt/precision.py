# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-module dtype/precision overrides for Gemma4.

Reads precision_overrides.json and resolves a (model variant, mesh shape) tuple
into a {module_name: ttnn.DataType} mapping. Modules without an override use a
caller-supplied default (typically bfloat16).

The JSON file is the single source of truth for per-system precision tweaks
(e.g. dropping shared_mlp to bfp8 on Gemma4-31B at 1x2 to fit DRAM). New entries
are added there rather than in code.
"""

import json
import os

import ttnn

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precision_overrides.json")

# Module names that may be overridden — keep in sync with the JSON schema and
# with the constructors that accept these kwargs (Gemma4Model and below).
# ``shared_mlp_down`` overrides only the MLP down_proj (the row-parallel
# reduction, the most bfp4-sensitive weight); it defaults to whatever
# ``shared_mlp`` resolves to, so gate/up can go bfp4 while down stays bfp8.
KNOWN_MODULES = ("shared_mlp", "shared_mlp_down", "attention", "experts", "router", "lm_head", "embedding")

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
    """Short stable string for cache-filename suffixes ("bf16" / "bfp8" / "fp32").

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

    @staticmethod
    def model_key_from_path(model_path):
        """Derive the JSON model key from a checkpoint path.

        Handles both plain repo dirs (``.../gemma-4-31B-it``) and HF cache
        snapshot layouts (``.../models--google--gemma-4-31B-it/snapshots/<rev>``),
        where the basename is the revision (e.g. "main") rather than the model
        name. In the cache layout we recover the name from the ``models--org--name``
        directory (the segment after the last ``--``).
        """
        path = str(model_path).rstrip("/")
        parts = path.split("/")
        try:
            snap_idx = parts.index("snapshots")
            slug = parts[snap_idx - 1]
            if slug.startswith("models--") and "--" in slug:
                return slug.split("--")[-1]
        except (ValueError, IndexError):
            pass
        return os.path.basename(path)

    @classmethod
    def load(cls, model_path, mesh_shape):
        """Resolve overrides for the given (model, mesh).

        model_path: full path to the HF checkpoint; we key on the model name.
        mesh_shape: (rows, cols) tuple, formatted as "RxC" for the JSON key.
        """
        model_key = cls.model_key_from_path(model_path)
        mesh_key = f"{mesh_shape[0]}x{mesh_shape[1]}"

        try:
            with open(_PATH) as f:
                table = json.load(f)
        except FileNotFoundError:
            return cls({})

        model_entry = table.get(model_key)
        if not model_entry:
            return cls({})

        # Mesh-specific override wins over "default"
        raw = model_entry.get(mesh_key) or model_entry.get("default") or {}
        resolved = {}
        for k, v in raw.items():
            if k not in KNOWN_MODULES:
                continue  # ignore unknown / future keys silently
            if v not in _DTYPE_BY_NAME:
                raise ValueError(
                    f"precision_overrides.json[{model_key}][{mesh_key}][{k}]={v!r} — "
                    f"unknown dtype; expected one of {sorted(_DTYPE_BY_NAME)}"
                )
            resolved[k] = _DTYPE_BY_NAME[v]

        # Runtime env overrides win over the JSON so a bfp4 A/B (Phase 2) can be
        # toggled without editing the checked-in baseline — e.g.
        #   GEMMA4_PRECISION_SHARED_MLP=bfp4
        #   GEMMA4_PRECISION_LM_HEAD=bfp4
        #   GEMMA4_PRECISION_ALL=bfp4   (applies to every known module)
        # Cache filenames embed the dtype, so bfp8 and bfp4 weights coexist on
        # disk and switching precisions never reuses a stale cache.
        resolved.update(cls._env_overrides())
        return cls(resolved)

    @staticmethod
    def _env_overrides():
        """Collect per-module dtype overrides from the environment.

        ``GEMMA4_PRECISION_ALL`` sets every known module; per-module
        ``GEMMA4_PRECISION_<MODULE>`` (e.g. ``GEMMA4_PRECISION_SHARED_MLP``) wins
        over ``_ALL``. Unknown dtype strings raise so a typo fails loudly.
        """
        out = {}
        all_val = os.environ.get("GEMMA4_PRECISION_ALL")
        if all_val is not None:
            if all_val not in _DTYPE_BY_NAME:
                raise ValueError(f"GEMMA4_PRECISION_ALL={all_val!r} — expected one of {sorted(_DTYPE_BY_NAME)}")
            for mod in KNOWN_MODULES:
                out[mod] = _DTYPE_BY_NAME[all_val]
        for mod in KNOWN_MODULES:
            val = os.environ.get(f"GEMMA4_PRECISION_{mod.upper()}")
            if val is not None:
                if val not in _DTYPE_BY_NAME:
                    raise ValueError(
                        f"GEMMA4_PRECISION_{mod.upper()}={val!r} — expected one of {sorted(_DTYPE_BY_NAME)}"
                    )
                out[mod] = _DTYPE_BY_NAME[val]
        return out
