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

import os

ENV = "TT_PERF_LAYERS"


def set_depth(env, depth) -> dict:
    """Express `depth` to a model builder through the mapping `env`.

    A positive int caps the build to that many blocks. ANY non-positive or unparseable depth --
    including None and 0 -- means ALL LAYERS and is expressed by DELETING the variable, never by
    writing a sentinel a builder could read as a count.
    """
    try:
        d = int(depth)
    except (TypeError, ValueError):
        d = 0
    if d > 0:
        env[ENV] = str(d)
    else:
        env.pop(ENV, None)
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
