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
import re

import ttnn

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precision_overrides.json")

# Module names that may be overridden — keep in sync with the JSON schema and
# with the constructors that accept these kwargs (Gemma4Model and below).
KNOWN_MODULES = ("shared_mlp", "attention", "experts", "router", "lm_head", "embedding")

_DTYPE_BY_NAME = {
    "bf16": ttnn.bfloat16,
    "bfloat16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfloat8_b": ttnn.bfloat8_b,
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
    if dtype == ttnn.float32:
        return "fp32"
    raise ValueError(f"No cache-suffix mapping for dtype {dtype}")


class Gemma4Precision:
    """Per-module dtype mapping. Construct via ``Gemma4Precision.load(...)``
    or directly with ``Gemma4Precision({...})``."""

    def __init__(self, overrides=None, single_tile_dest_acc=True, ccl_topology=None):
        self._overrides = dict(overrides) if overrides else {}
        # Whether the m<=32 projections accumulate in fp32. Model-wide, not
        # per-module or per-mesh, and deliberately not keyed on the weight
        # dtype: 12B and E2B want opposite answers on the same all-bfp8
        # weights. Measurements and rationale live with the decision, in
        # ``single_tile_matmul_ckc``.
        self.single_tile_dest_acc = bool(single_tile_dest_acc)
        # Forced CCL topology, or None to let default_ccl_topology pick from the
        # arch. Model-wide like single_tile_dest_acc: Ring and Linear reduce in
        # different orders, so this is a numerics knob, not a perf-only one.
        self.ccl_topology = ccl_topology

    def get(self, module_name, default=ttnn.bfloat16):
        return self._overrides.get(module_name, default)

    def __repr__(self):
        return (
            f"Gemma4Precision({self._overrides!r}, single_tile_dest_acc={self.single_tile_dest_acc}, "
            f"ccl_topology={self.ccl_topology!r})"
        )

    @classmethod
    def load(cls, model_path, mesh_shape):
        """Resolve overrides for the given (model, mesh).

        model_path: full path to the HF checkpoint; we key on the basename.
        mesh_shape: (rows, cols) tuple, formatted as "RxC" for the JSON key.
        """
        path = str(model_path).rstrip("/")
        model_key = os.path.basename(path)
        # Under HF_HUB_OFFLINE vLLM replaces the repo id with the resolved
        # snapshot directory (.../models--{org}--{name}/snapshots/{hash}), so
        # the basename is the snapshot hash and the variant lookup silently
        # misses every override (31B then loads all-bf16: +~7.9 GB/chip at
        # tp=4, which OOM'd the QB2 vLLM CI cell at 256k context). Recover the
        # repo basename from the hub layout.
        hub_match = re.search(r"models--[^/]+--([^/]+)/snapshots/[^/]+$", path)
        if hub_match:
            model_key = hub_match.group(1)
        mesh_key = f"{mesh_shape[0]}x{mesh_shape[1]}"

        try:
            with open(_PATH) as f:
                table = json.load(f)
        except FileNotFoundError:
            return cls({})

        model_entry = table.get(model_key)
        if not model_entry:
            return cls({})

        # Model-wide (not per-mesh): read off the model entry, not the mesh
        # sub-dict, because a mesh entry REPLACES "default" rather than merging.
        dest_acc = model_entry.get("single_tile_dest_acc", True)
        if isinstance(dest_acc, dict):
            # Per-arch form, for a flag that is a workaround rather than a
            # preference: an arch the object does not name keeps the default.
            # 31B needs it because its reason (Wormhole #38306, HiFi3 with fp32
            # dest-accumulation) is a Wormhole hardware bug, and applying the
            # workaround model-wide changed Blackhole too, where main runs and
            # validates the default.
            unknown = sorted(set(dest_acc) - set(_ARCH_KEYS))
            if unknown:
                raise ValueError(
                    f"precision_overrides.json[{model_key}][single_tile_dest_acc] has unknown arch "
                    f"key(s) {unknown} — expected one of {sorted(_ARCH_KEYS)}"
                )
            dest_acc = dest_acc.get(_current_arch_key(), True)
        if not isinstance(dest_acc, bool):
            raise ValueError(
                f"precision_overrides.json[{model_key}][single_tile_dest_acc]={dest_acc!r} — "
                f"expected true/false, or an object keyed by {sorted(_ARCH_KEYS)}"
            )

        topology = model_entry.get("ccl_topology")
        if topology is not None and topology not in ("ring", "linear"):
            raise ValueError(
                f"precision_overrides.json[{model_key}][ccl_topology]={topology!r} — expected 'ring' or 'linear'"
            )

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
        return cls(resolved, single_tile_dest_acc=dest_acc, ccl_topology=topology)


_ARCH_KEYS = ("wormhole_b0", "blackhole")


def _current_arch_key():
    """Arch key for per-arch override objects."""
    from models.common.utility_functions import is_blackhole

    return "blackhole" if is_blackhole() else "wormhole_b0"


_DEST_ACC_BY_MODEL = {}


def default_single_tile_dest_acc():
    """Per-process ``single_tile_dest_acc`` for callers without a model path.

    Gemma4Model threads the resolved flag down explicitly, but the unit tests
    build Gemma4Attention / SharedMLP straight from an HF config, so nothing
    reaches them that way. They select the variant exactly as the rest of the
    suite does -- HF_MODEL -- so fall back to that. An explicitly passed value
    always wins over this.
    """
    key = os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH") or ""
    if key not in _DEST_ACC_BY_MODEL:
        _DEST_ACC_BY_MODEL[key] = Gemma4Precision.load(key, (1, 1)).single_tile_dest_acc if key else True
    return _DEST_ACC_BY_MODEL[key]
