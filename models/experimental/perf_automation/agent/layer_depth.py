"""The ONE place that expresses "how deep should this model build" to a model's own builder.

WHY THIS EXISTS (2026-07-26)
    The tool used to say "all layers" by sending ``TT_PERF_LAYERS=0``. Three separate pieces of
    generated model code then read that as "build ZERO layers", because the value arrives as the
    STRING "0" and the natural guard is truthiness:

        _perf = os.environ.get("TT_PERF_LAYERS")        # "0" -- a non-empty string, so TRUTHY
        num_layers = int(_perf) if _perf else None      # int("0") == 0  ->  zero layers

    A zero-layer model has no KV cache, so it died in ``get_block_size(kv_cache[0][0])`` before
    emitting any timing marker. The full-pipeline gate could only report "no markers", and the
    correctness gate was computing PCC against a model that had done nothing. It cost a day, and it
    was authored three times, because ``0`` is indistinguishable from a legitimate layer count.

THE FIX IS THE ABSENCE OF A VALUE
    "All layers" is now expressed by REMOVING the variable, not by any sentinel. That makes the
    idiom above CORRECT BY ACCIDENT: ``os.environ.get`` returns None, the guard is falsy, and the
    builder takes its own all-layers branch. There is no value left that a builder can misread,
    because there is no value.

    A positive integer still means "cap the profiled window to this many blocks". Nothing else is a
    legal depth: 0, negative numbers and junk all mean ALL LAYERS, since none of them is a depth a
    caller could sensibly want.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ENV = "TT_PERF_LAYERS"

# Keys that declare "how many repeated blocks does this model have", across the config dialects
# tt-metal models actually ship: HF transformers, Meta llama params.json, and hand-rolled JSON.
_DEPTH_KEYS = (
    "num_hidden_layers",  # HF transformers (llama, mistral, qwen, ...)
    "n_layers",  # Meta llama params.json
    "num_layers",
    "n_layer",  # GPT-2 lineage
    "num_blocks",
    "num_decoder_layers",
    "decoder_layers",
    "gpt_layers",  # XTTS
    "depth",
)

# Directories that hold OTHER models' configs. Never descend into these: a llama demo carries
# model_params/Qwen2.5-VL-72B-Instruct/config.json and ~40 more, so a recursive scan silently
# returns a different model's depth -- the same class of wrong-number-looks-plausible bug this
# module exists to stop.
_FOREIGN_DIRS = ("model_params", "reference_outputs", "sweeps", "model_cache", "tests")


def _depth_from_mapping(obj) -> int | None:
    """The first recognised depth key with a positive int value, or None."""
    if not isinstance(obj, dict):
        return None
    for key in _DEPTH_KEYS:
        v = obj.get(key)
        if isinstance(v, bool):
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    for nested in ("text_config", "decoder", "model", "gpt", "llm_config"):
        n = _depth_from_mapping(obj.get(nested))
        if n is not None:
            return n
    return None


def full_depth_from_config(model_id: str = "", model_dir=None) -> int | None:
    """How many repeated blocks does this model have, read WITHOUT building or running it.

    Resolution order, most authoritative first:
      1. HF transformers config for `model_id` (also covers custom architectures via
         trust_remote_code, and nested text_config for multimodal wrappers).
      2. A config file sitting at the ROOT of `model_dir` -- config.json / params.json /
         model_config.json -- for models that ship their own, HF or not.

    Returns None rather than a guess when nothing declares it, so the caller falls back to letting
    the builder reveal its own depth. Never recurses: see _FOREIGN_DIRS for why.
    """
    if model_id:
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(str(model_id), trust_remote_code=True)
            n = _depth_from_mapping(getattr(cfg, "__dict__", {}) or {})
            if n is not None:
                return n
        except Exception:  # noqa: BLE001
            pass
    if model_dir:
        root = Path(model_dir)
        for name in ("config.json", "params.json", "model_config.json"):
            p = root / name
            if not p.is_file():
                continue
            try:
                n = _depth_from_mapping(json.loads(p.read_text(errors="ignore")))
            except Exception:  # noqa: BLE001
                continue
            if n is not None:
                return n
    return None


FORCE_ALL = "PERF_MCP_FORCE_ALL_LAYERS"


def set_depth(env, depth, key: str | None = None) -> dict:
    """Express `depth` to a model builder through the mapping `env`.

    `key` names the variable to write, defaulting to this tool's own ENV. A model that exposes its
    OWN depth variable still needs the same two-part convention, and hardcoding ENV here was why
    three call sites wrote their discovered knob raw instead (leaving FORCE_ALL in whatever state
    it happened to be in).

    A positive int caps the build to that many blocks. ANY non-positive or unparseable depth --
    including None and 0 -- means ALL LAYERS and is expressed by DELETING the variable, never by
    writing a sentinel a builder could read as a count.

    Asking for ALL layers also arms the depth guard (PERF_MCP_FORCE_ALL_LAYERS=1), because absence
    alone is not enough: a perf test can fill the cap back in at import with
    os.environ.setdefault(...), silently turning "whole model" into a 2-layer build. The flag is set
    HERE rather than at each call site so no caller can express "all layers" and forget to defend it;
    the guard itself only acts if the invocation also loads agent/depth_guard_plugin via `-p`.
    Requesting a positive cap clears the flag, so the tracy slice is never stripped.
    """
    var = key or ENV
    try:
        d = int(depth)
    except (TypeError, ValueError):
        d = 0
    if d > 0:
        env[var] = str(d)
        env.pop(FORCE_ALL, None)
    else:
        env.pop(var, None)
        env[FORCE_ALL] = "1"
    return env


def read_depth(environ=None):
    """The depth a builder should use: a positive int, or None meaning ALL LAYERS.

    None is the sentinel every builder already understands for "no cap", so a caller can pass this
    straight through to its factory without re-deriving the convention.
    """
    src = os.environ if environ is None else environ
    raw = str(src.get(ENV) or "").strip()
    try:
        d = int(raw)
    except ValueError:
        return None
    return d if d > 0 else None
