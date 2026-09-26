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


_ARCH_KEYS = ("wormhole_b0", "blackhole")


def _current_arch_key():
    """Arch key for per-arch override objects."""
    # Imported per call on purpose: test_precision_overrides monkeypatches
    # ``models.common.utility_functions.is_blackhole``, which a module-level
    # import would not see.
    from models.common.utility_functions import is_blackhole

    return "blackhole" if is_blackhole() else "wormhole_b0"


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

    def __init__(self, overrides=None, single_tile_dest_acc=DEFAULT_SINGLE_TILE_DEST_ACC, ccl_topology=None):
        self._overrides = dict(overrides) if overrides else {}
        # fp32 destination accumulation on the m<=32 projections. Per model, not
        # global: it is what carries 12B's accuracy, and it is what collapses
        # 31B's 128k decode into a repetition loop. See single_tile_matmul_ckc.
        self.single_tile_dest_acc = bool(single_tile_dest_acc)
        # Forced CCL topology ("ring" / "linear"), or None to let
        # default_ccl_topology pick from the arch. Model-wide like
        # single_tile_dest_acc: Ring and Linear reduce in different orders, so
        # this is a numerics knob, not a perf-only one.
        self.ccl_topology = ccl_topology

    def get(self, module_name, default=ttnn.bfloat16):
        return self._overrides.get(module_name, default)

    def __repr__(self):
        return (
            f"Gemma4Precision({self._overrides!r}, single_tile_dest_acc={self.single_tile_dest_acc}, "
            f"ccl_topology={self.ccl_topology!r})"
        )

    @classmethod
    def load(cls, model_path, mesh_shape, max_seq_len=None):
        """Resolve overrides for the given (model, mesh).

        model_path: full path to the HF checkpoint; we key on the basename.
        mesh_shape: (rows, cols) tuple, formatted as "RxC" for the JSON key.
        max_seq_len: served context. bfp8 modules are downgraded to bf16 above
            the variant's ``bfp8_max_context`` (see below).
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
        # MESH-SCOPED form: {"1x8": {...}, "default": {...}}, told apart from the
        # flat {module: limit} form by its keys being mesh shapes. The ceiling
        # trades memory for coherence -- downgrading to bf16 DOUBLES those
        # weights -- so it can only be declared where bf16 actually fits. It was
        # measured on tp=8; applying it to every mesh hung Gemma4-31B on
        # bh_quietbox_2 (1x4) in model init, because 31B shared_mlp in bf16 at
        # 262144 does not fit on four chips. A mesh with no entry keeps bfp8,
        # which is what main did before the ceiling existed.
        if isinstance(limits, dict) and limits and all(re.fullmatch(r"\d+x\d+|default", str(k)) for k in limits):
            limits = limits.get(mesh_key, limits.get("default"))
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
        if isinstance(dest_acc, dict):
            # Per-arch form, for a flag that is a workaround rather than a
            # preference: an arch the object does not name keeps the default.
            # 31B needs it because its reason -- Wormhole #38306, HiFi3 paired
            # with fp32 dest-accumulation -- is a Wormhole hardware bug, and
            # writing the workaround model-wide also turned it off on Blackhole,
            # where main runs 31B with the default and passes.
            unknown = sorted(set(dest_acc) - set(_ARCH_KEYS))
            if unknown:
                raise ValueError(
                    f"precision_overrides.json[{model_key}][single_tile_dest_acc] has unknown arch "
                    f"key(s) {unknown} — expected one of {sorted(_ARCH_KEYS)}"
                )
            dest_acc = dest_acc.get(_current_arch_key(), DEFAULT_SINGLE_TILE_DEST_ACC)
        if not isinstance(dest_acc, bool):
            raise ValueError(
                f"precision_overrides.json[{model_key}][single_tile_dest_acc]={dest_acc!r} — "
                f"expected true or false, or an object keyed by {sorted(_ARCH_KEYS)}"
            )
        # Same model-wide argument as single_tile_dest_acc above.
        if "ccl_topology" in raw:
            raise ValueError(
                f"precision_overrides.json[{model_key}][{mesh_key}][ccl_topology] — "
                "this flag is model-wide; put it on the model entry, not a mesh entry"
            )
        topology = model_entry.get("ccl_topology")
        if topology is not None and topology not in ("ring", "linear"):
            raise ValueError(
                f"precision_overrides.json[{model_key}][ccl_topology]={topology!r} — expected 'ring' or 'linear'"
            )
        return cls(resolved, single_tile_dest_acc=dest_acc, ccl_topology=topology)


def default_single_tile_dest_acc():
    """Fallback dest-accumulation policy, resolved from the environment.

    Gemma4Model threads ``precision.single_tile_dest_acc`` down instead, so a run
    selected by ``create_tt_model(model_path=...)`` reads it from the same
    explicit checkpoint as the dtype overrides; this serves the unit tests, which
    build the modules from an HF config with nothing threaded through. The env
    chain matches ``create_tt_model``'s: reading only HF_MODEL returned the
    default for a run selected with GEMMA4_MODEL_PATH, which on 31B is the fp32
    dest-accumulation its long generation degenerates under. The mesh shape is
    arbitrary; ``load`` reads the flag off the model entry, which no
    mesh-specific block can shadow.
    """
    model_path = os.environ.get("HF_MODEL") or os.environ.get("GEMMA4_MODEL_PATH")
    if not model_path:
        return DEFAULT_SINGLE_TILE_DEST_ACC
    return Gemma4Precision.load(model_path, (1, 1)).single_tile_dest_acc


def resolve_single_tile_dest_acc(threaded=None):
    """The threaded policy when a caller passed one, else the env fallback."""
    return default_single_tile_dest_acc() if threaded is None else bool(threaded)
