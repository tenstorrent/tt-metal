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

from loguru import logger

import ttnn

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precision_overrides.json")

# Module names that may be overridden — keep in sync with the JSON schema and
# with the constructors that accept these kwargs (Gemma4Model and below).
KNOWN_MODULES = ("shared_mlp", "attention", "experts", "router", "lm_head", "embedding")


def _env_overrides():
    """Per-module dtype overrides from ``GEMMA4_PRECISION_OVERRIDE``, or ``{}``.

    Format is ``module=dtype`` pairs, comma separated:

        GEMMA4_PRECISION_OVERRIDE=attention=bf16,shared_mlp=bf16

    These win over precision_overrides.json. The point is A/B-ability: module
    dtype is one of the biggest levers on accuracy, and until now changing it
    meant editing a checked-in JSON — which makes a sweep awkward, leaves no
    record in the run's own log of what was tested, and is exactly how earlier
    A/B pairs ended up bit-identical because the intended change never took
    effect. Score any sweep on the per-layer PCC ladder against the HF reference,
    not on end-to-end token counts.

    Unknown module names and bad dtypes raise rather than being ignored: a typo'd
    knob that silently does nothing is the failure mode this exists to prevent.
    """
    raw = os.environ.get("GEMMA4_PRECISION_OVERRIDE", "").strip()
    if not raw:
        return {}
    out = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"GEMMA4_PRECISION_OVERRIDE: expected 'module=dtype' pairs, got {item!r}")
        name, value = (part.strip() for part in item.split("=", 1))
        if name not in KNOWN_MODULES:
            raise ValueError(f"GEMMA4_PRECISION_OVERRIDE: unknown module {name!r}; expected one of {KNOWN_MODULES}")
        if value not in _DTYPE_BY_NAME:
            raise ValueError(
                f"GEMMA4_PRECISION_OVERRIDE: unknown dtype {value!r} for {name}; "
                f"expected one of {sorted(_DTYPE_BY_NAME)}"
            )
        out[name] = _DTYPE_BY_NAME[value]
    return out


_DTYPE_BY_NAME = {
    "bf16": ttnn.bfloat16,
    "bfloat16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "fp32": ttnn.float32,
    "float32": ttnn.float32,
}


def _model_key_candidates(model_path, hf_config=None):
    """Table-key candidates for the active checkpoint, best first.

    The basename alone is not enough. ``HF_MODEL`` is commonly a lowercased HF
    id (``google/gemma-4-31b-it`` — what the Tracy profile docstrings tell you
    to export) or a hashed snapshot dir under ``~/.cache/huggingface/hub/.../
    snapshots/<sha>``. Both miss the canonical ``gemma-4-31B-it`` key, and a
    miss silently downgrades every module to bf16. So fall back to identifying
    the variant from the config, the same way ``tests/test_factory.py``
    resolves its PCC-threshold keys.
    """
    candidates = [os.path.basename(str(model_path).rstrip("/"))]

    config = hf_config
    if config is None:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            # Config inference is best-effort; the basename may still match.
            return tuple(dict.fromkeys(candidates))

    text_config = getattr(config, "text_config", config)
    hidden = getattr(text_config, "hidden_size", None)
    is_moe = bool(getattr(text_config, "enable_moe_block", False))
    if is_moe:
        candidates.append("gemma-4-26B-A4B-it")
    elif hidden == 5376:
        candidates.append("gemma-4-31B-it")
    elif hidden == 3840:
        candidates.append("gemma-4-12B-it")
    return tuple(dict.fromkeys(candidates))


def _lookup_model_entry(table, candidates):
    """First (key, entry) in ``candidates`` present in ``table``, case-insensitively.

    Returns ``(None, None)`` when nothing matches. Case folding matters because
    HF ids lowercase the variant (``gemma-4-31b-it``) while the table keys use
    the checkpoint's own casing (``gemma-4-31B-it``).
    """
    by_lower = {k.lower(): (k, v) for k, v in table.items()}
    for candidate in candidates:
        if candidate in table:
            return candidate, table[candidate]
        hit = by_lower.get(candidate.lower())
        if hit:
            return hit
    return None, None


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
    def load(cls, model_path, mesh_shape, hf_config=None):
        """Resolve overrides for the given (model, mesh).

        model_path: full path to the HF checkpoint, or an HF id; the basename is
            the first key candidate. ``hf_config`` (when passed) supplies the
            canonical-variant fallback for paths the basename can't identify —
            see ``_model_key_candidates``.
        mesh_shape: (rows, cols) tuple, formatted as "RxC" for the JSON key.
        """
        mesh_key = f"{mesh_shape[0]}x{mesh_shape[1]}"

        try:
            with open(_PATH) as f:
                table = json.load(f)
        except FileNotFoundError:
            return cls(_env_overrides())

        candidates = _model_key_candidates(model_path, hf_config)
        model_key, model_entry = _lookup_model_entry(table, candidates)
        if not model_entry:
            # A silent miss here downgrades every module to the caller's default
            # dtype (bf16), which looks like a perf regression with no error
            # — so say so loudly. Only warn when the table actually has entries
            # to match against (an empty/absent table is a valid "no overrides").
            if any(k for k in table if not k.startswith("_")):
                logger.warning(
                    "Gemma4 precision: no precision_overrides.json entry for any of {} "
                    "(table has {}); every module falls back to the caller's default dtype.",
                    list(candidates),
                    sorted(k for k in table if not k.startswith("_")),
                )
            return cls(_env_overrides())

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
        logger.info(
            "Gemma4 precision: resolved {}[{}] -> {}",
            model_key,
            mesh_key,
            {k: v for k, v in raw.items() if k in KNOWN_MODULES},
        )
        env = _env_overrides()
        if env:
            logger.warning(
                "Gemma4 precision: GEMMA4_PRECISION_OVERRIDE applied on top of the table -> {}",
                {k: dtype_to_str(v) for k, v in env.items()},
            )
            resolved.update(env)
        return cls(resolved)
