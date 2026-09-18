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
from models.common.weight_cache import checkpoint_name

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precision_overrides.json")

# Module names that may be overridden — keep in sync with the JSON schema and
# with the constructors that accept these kwargs (Gemma4Model and below).
KNOWN_MODULES = ("shared_mlp", "attention", "experts", "router", "lm_head", "embedding")

# Non-dtype, model-wide numerics flags that live in the same table. They are not
# module dtypes, so they are read separately from the KNOWN_MODULES loop.
DEFAULT_SINGLE_TILE_DEST_ACC = True

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

    def __init__(self, overrides=None, single_tile_dest_acc=DEFAULT_SINGLE_TILE_DEST_ACC):
        self._overrides = dict(overrides) if overrides else {}
        # fp32 destination accumulation on the m<=32 projections. Per model, not
        # global: it is what carries 12B's accuracy, and it is what collapses
        # 31B's 128k decode into a repetition loop. See single_tile_matmul_ckc.
        self.single_tile_dest_acc = bool(single_tile_dest_acc)

    def get(self, module_name, default=ttnn.bfloat16):
        return self._overrides.get(module_name, default)

    def __repr__(self):
        return f"Gemma4Precision({self._overrides!r})"

    @classmethod
    def load(cls, model_path, mesh_shape):
        """Resolve overrides for the given (model, mesh).

        model_path: full path to the HF checkpoint; we key on the basename.
        mesh_shape: (rows, cols) tuple, formatted as "RxC" for the JSON key.
        """
        # Under HF_HUB_OFFLINE vLLM replaces the repo id with the resolved
        # snapshot directory (.../models--{org}--{name}/snapshots/{hash}); a
        # plain basename would be the snapshot hash and the variant lookup
        # would silently miss every override (31B then loads all-bf16:
        # +~7.9 GB/chip at tp=4, which OOM'd the QB2 vLLM CI cell at 256k
        # context). checkpoint_name() recovers the repo basename from the hub
        # layout; the warm weight-cache identity in common.py uses the same
        # helper so both key on one name.
        model_key = checkpoint_name(model_path)
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
        # Read from the MODEL entry, not the mesh dict. A mesh entry replaces
        # "default" wholesale (see the lookup above), so a per-mesh flag would
        # silently revert to the default the moment someone adds a mesh-specific
        # dtype block -- and this flag going quietly back to true is a 128k
        # repetition loop on 31B. Model-wide is also what makes
        # default_single_tile_dest_acc()'s fixed (1, 1) lookup correct.
        if "single_tile_dest_acc" in raw:
            raise ValueError(
                f"precision_overrides.json[{model_key}][{mesh_key}][single_tile_dest_acc] — "
                "this flag is model-wide; put it on the model entry, not a mesh entry"
            )
        dest_acc = model_entry.get("single_tile_dest_acc", DEFAULT_SINGLE_TILE_DEST_ACC)
        if not isinstance(dest_acc, bool):
            raise ValueError(
                f"precision_overrides.json[{model_key}][single_tile_dest_acc]=" f"{dest_acc!r} — expected true or false"
            )
        return cls(resolved, single_tile_dest_acc=dest_acc)


def default_single_tile_dest_acc():
    """The variant's dest-accumulation policy, resolved from HF_MODEL.

    Gemma4Attention / SharedMLP are constructed straight from an HF config by
    the unit tests, so a value threaded through Gemma4Model never reaches them.
    Both resolve the policy here instead, from the same table and the same
    checkpoint name the rest of the precision lookup uses. The mesh shape passed
    here is arbitrary because ``load`` reads this flag off the model entry, which
    no mesh-specific block can shadow.
    """
    model_path = os.environ.get("HF_MODEL")
    if not model_path:
        return DEFAULT_SINGLE_TILE_DEST_ACC
    return Gemma4Precision.load(model_path, (1, 1)).single_tile_dest_acc
