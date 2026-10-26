# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill profiling zones (Tracy signposts), off unless ``MIMO_PROFILE_ZONES=1``.

A zone is two signposts, ``MIMO_ZONE_START <name>`` and ``MIMO_ZONE_END <name>``. They land in the tracy ops CSV
in host-enqueue order between the op rows, so the ops between a zone's two signposts are the ops it enqueued
(tests/perf/parse_prefill_zones.py sums their device time). Signposts are host messages: no device op, no sync.
Same scheme as models/demos/minimax_m3/utils/profiler_utils.py, whose parser this reuses.

    with zone("attn"):
        with steps() as step:
            step("qkv")   # ends the previous step, starts "qkv"
            ...
            step("sdpa")
            ...
"""

import contextlib
import os

import ttnn

ZONE_START = "MIMO_ZONE_START"
ZONE_END = "MIMO_ZONE_END"
ENABLED = os.environ.get("MIMO_PROFILE_ZONES", "0") == "1"


def _signpost(text):
    # tracy.signpost's wire format, without its log line per call (a chunk opens ~20 zones per layer).
    ttnn.tracy_message(f"`TT_SIGNPOST: {text}`")


@contextlib.contextmanager
def _zone(name):
    _signpost(f"{ZONE_START} {name}")
    try:
        yield
    finally:
        _signpost(f"{ZONE_END} {name}")


@contextlib.contextmanager
def _steps():
    current = []

    def step(name):
        if current:
            _signpost(f"{ZONE_END} {current.pop()}")
        current.append(name)
        _signpost(f"{ZONE_START} {name}")

    try:
        yield step
    finally:
        if current:
            _signpost(f"{ZONE_END} {current.pop()}")


_NULL = contextlib.nullcontext()
_NULL_STEPS = contextlib.nullcontext(lambda name: None)


def zone(name):
    """Context manager: one zone around the ops enqueued inside it."""
    return _zone(name) if ENABLED else _NULL


def steps():
    """Context manager yielding ``step(name)``: ends the current step's zone and starts zone ``name``; the last one
    ends on exit."""
    return _steps() if ENABLED else _NULL_STEPS
