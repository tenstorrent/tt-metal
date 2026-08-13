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


def _walk_depths(obj, _seen=None, _out=None) -> list:
    """EVERY declared depth anywhere in a config, whatever shape it is nested in.

    Three containers, one rule, because configs mix all of them:

        dict     -> its values          plain JSON
        object   -> its __dict__        transformers PretrainedConfig
        list     -> its items           sub_configs = [text_cfg, vision_cfg]

    A single named walk over ("text_config", "decoder", ...) could not see any of this. It stopped at
    Gemma3Config.text_config -- an OBJECT, not a dict -- so gemma-3-12b-it declared no depth at all,
    nothing clamped the coverage window, and a 48-layer model was profiled and reported as 96.

    Returns every candidate rather than the first, because the first is a coin flip: gemma3 offers
    text_config=48 and vision_config=27 and the walk order decides which one you get. The caller
    takes the MAX, which is the safe direction for a value used only as a ceiling -- too high just
    weakens the clamp, too low silently hides layers.

    _seen is termination, not a depth limit: configs hold back-references and the walk is otherwise
    unbounded by design.
    """
    _seen = set() if _seen is None else _seen
    _out = [] if _out is None else _out
    if id(obj) in _seen:
        return _out
    if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
        return _out
    _seen.add(id(obj))

    if isinstance(obj, (list, tuple)):
        for item in obj:
            _walk_depths(item, _seen, _out)
        return _out

    if not isinstance(obj, dict):
        d = getattr(obj, "__dict__", None)
        if not isinstance(d, dict):
            return _out
        obj = d
        if id(obj) in _seen:
            return _out
        _seen.add(id(obj))

    for key in _DEPTH_KEYS:
        v = obj.get(key)
        if isinstance(v, bool):
            # True is an int in Python; num_hidden_layers=True must not become a 1-layer window.
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            _out.append(n)
    for v in obj.values():
        _walk_depths(v, _seen, _out)
    return _out


def _depth_from_mapping(obj) -> int | None:
    """The DEEPEST declared depth anywhere in `obj`, or None when nothing declares one."""
    found = _walk_depths(obj)
    return max(found) if found else None


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
        # EVERY config at the root, not the first that parses. A model that ships both a params.json
        # and a config.json declares the same thing twice and the shallower file must not win by
        # filename order -- same max rule as within one file.
        seen = []
        for name in ("config.json", "params.json", "model_config.json"):
            p = root / name
            if not p.is_file():
                continue
            try:
                n = _depth_from_mapping(json.loads(p.read_text(errors="ignore")))
            except Exception:  # noqa: BLE001
                continue
            if n is not None:
                seen.append(n)
        if seen:
            return max(seen)
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


def declared_section_depths(model_id: str = "", model_dir=None) -> list:
    """EVERY declared block depth, per section, without collapsing to one number.

    THE INFORMATION WAS ALWAYS COLLECTED AND ALWAYS DISCARDED. _walk_depths returns every depth a
    config declares -- for Voxtral-Mini-3B that is the audio tower's 32 AND the text decoder's 32 --
    and _depth_from_mapping reduces it with max() one line later, because both of its callers wanted
    a single ceiling.

    Multi-section models need the list. Voxtral has three sections and two independent stacks; the
    tool sized ONE depth, capped the text decoder to 2 and left both encoders at 32, and nothing
    could notice because "how many sections does this model have" was never asked. The config
    answers it for free, before the device is touched: no markers, no walk, no naming convention,
    and no per-model code -- transformers already parsed it.

    Sorted descending so the caller reads the deepest first; empty when nothing declares a depth.
    """
    if model_id:
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(str(model_id), trust_remote_code=True)
            found = _walk_depths(getattr(cfg, "__dict__", {}) or {})
            if found:
                return sorted(found, reverse=True)
        except Exception:  # noqa: BLE001
            pass
    if model_dir:
        from pathlib import Path as _P

        root = _P(model_dir)
        for name in ("config.json", "params.json", "model_config.json"):
            p = root / name
            if not p.is_file():
                continue
            try:
                import json as _j

                found = _walk_depths(_j.loads(p.read_text()))
                if found:
                    return sorted(found, reverse=True)
            except Exception:  # noqa: BLE001
                continue
    return []
