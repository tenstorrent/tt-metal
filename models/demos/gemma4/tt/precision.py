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

from loguru import logger

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

    def __init__(self, overrides=None):
        self._overrides = dict(overrides) if overrides else {}

    def get(self, module_name, default=ttnn.bfloat16):
        return self._overrides.get(module_name, default)

    def __repr__(self):
        return f"Gemma4Precision({self._overrides!r})"

    @classmethod
    def load(cls, model_path, mesh_shape, max_seq_len=None):
        """Resolve overrides for the given (model, mesh).

        model_path: full path to the HF checkpoint; we key on the basename.
        mesh_shape: (rows, cols) tuple, formatted as "RxC" for the JSON key.
        max_seq_len: served context. bfp8 modules are downgraded to bf16 above
            the variant's ``bfp8_max_context`` (see below).
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

        # Per-module context ceiling for bfp8. bfp8 error accumulates with
        # sequence length, but NOT uniformly across modules -- MEASURED on 31B /
        # tp=8 at a true 261,944-token prompt (same commit, build and prompt;
        # only the override differs):
        #
        #     attention + shared_mlp bfp8 : degenerate ("...laught laught...")
        #     shared_mlp bfp8 only        : degenerate ("...la la la la...")
        #     attention  bfp8 only        : COHERENT, 16.98 tok/s/u
        #     all bf16                    : COHERENT, 16.15 tok/s/u
        #
        # So shared_mlp is the module that cannot hold bfp8 at very long context,
        # and attention can -- keeping it quantized is both faster and closer to
        # the <=128k configuration. 128k is coherent with BOTH in bfp8, so the
        # ceiling sits between 131072 and 262144.
        #
        # This was invisible for months because a path-resolution bug (fixed in
        # a73264153281) made snapshot-style model paths miss the override table
        # entirely, so long-context runs silently used bf16 -- the banked
        # "coherent 256k" numbers were bf16 runs.
        #
        # ``bfp8_max_context`` accepts an int (applies to every bfp8 module) or a
        # {module: limit} dict. Downgrading (rather than raising) keeps long
        # context WORKING; GEMMA4_BFP8_MAX_CONTEXT overrides every limit, 0
        # disables the ceiling entirely.
        limits = model_entry.get("bfp8_max_context")
        env_limit = os.environ.get("GEMMA4_BFP8_MAX_CONTEXT")
        if env_limit is not None:
            try:
                limits = int(env_limit)
            except ValueError:
                limits = None
        if limits and max_seq_len:
            served = int(max_seq_len)
            downgraded = []
            for mod, dt in list(resolved.items()):
                if dt != ttnn.bfloat8_b:
                    continue
                lim = limits.get(mod) if isinstance(limits, dict) else limits
                if lim and served > int(lim):
                    resolved[mod] = ttnn.bfloat16
                    downgraded.append((mod, int(lim)))
            if downgraded:
                detail = ", ".join(f"{m} (>{l})" for m, l in sorted(downgraded))
                logger.warning(
                    f"Gemma4 precision: max_seq_len={served} exceeds the bfp8 context ceiling for "
                    f"{model_key}; downgrading {detail} bfp8 -> bf16 (bfp8 degenerates at very long "
                    "context). Costs memory/throughput; set GEMMA4_BFP8_MAX_CONTEXT=0 to disable."
                )
        return cls(resolved)
