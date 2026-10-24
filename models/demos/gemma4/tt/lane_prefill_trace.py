# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The traced lane prefill: which lane-frame shapes are captured at boot and when a call replays one.

A lane-frame prefill call is ``k`` rounds of one prompt per lane, each column prefilling its ``k`` users as one
stream of ``k x S`` tokens (``Gemma4Generator.prefill_forward_lanes``). The bucket ``kxS`` names one captured
device trace per mesh (GEMMA4_LANE_PREFILL_TRACE_BUCKETS, default ``1x128``: the served c1 frame, one real lane and
scratch lanes); a call whose rounds and padded length match a bucket replays it, every other call stays eager.
Leaf module: host-only arithmetic, importable under the unit conftest's stubs.
"""

import os

LANE_PREFILL_TRACE_ENV = "GEMMA4_LANE_PREFILL_TRACE"
LANE_PREFILL_TRACE_BUCKETS_ENV = "GEMMA4_LANE_PREFILL_TRACE_BUCKETS"
DEFAULT_BUCKETS = "1x128"
TRACE_SEQ_LENS = (128, 512, 1024, 2048, 4096)


def lane_prefill_trace_enabled():
    """GEMMA4_LANE_PREFILL_TRACE=1 (opt-in until the Galaxy window validates it): capture the lane-frame buckets at
    boot and replay them; off, every lane-frame call is eager."""
    return os.environ.get(LANE_PREFILL_TRACE_ENV, "0").lower() in ("1", "true", "yes")


def parse_lane_prefill_trace_buckets(raw, *, max_chunk, lane_slots, bounded_sliding):
    """``{(k, S), ...}`` from ``"1x128,2x128"``: k rounds per column times the padded length S. A bucket needs
    ``k * S`` within the prefill chunk, ``k`` within the lane's slots, ``S`` a trace ISL, and under bounded sliding
    ``k == 1`` (the kernel-capped ring fill carries ONE valid length per device: a round's single prompt length).
    Malformed or out-of-range entries are dropped (returned separately for the log)."""
    buckets, dropped = set(), []
    for token in str(raw or "").split(","):
        token = token.strip().lower()
        if not token:
            continue
        try:
            k, s = (int(x) for x in token.split("x"))
        except ValueError:
            dropped.append(token)
            continue
        ok = (
            k >= 1
            and k <= int(lane_slots)
            and s in TRACE_SEQ_LENS
            and k * s <= int(max_chunk)
            and (k == 1 or not bounded_sliding)
        )
        (buckets.add((k, s)) if ok else dropped.append(token))
    return buckets, dropped


def lane_prefill_trace_buckets(*, max_chunk, lane_slots, bounded_sliding, raw=None):
    """The configured buckets (empty when the trace is off)."""
    if not lane_prefill_trace_enabled():
        return set(), []
    raw = os.environ.get(LANE_PREFILL_TRACE_BUCKETS_ENV, DEFAULT_BUCKETS) if raw is None else raw
    return parse_lane_prefill_trace_buckets(
        raw, max_chunk=max_chunk, lane_slots=lane_slots, bounded_sliding=bounded_sliding
    )


def lane_prefill_trace_choice(captured, rounds, padded_len, *, single_chunk=True, cached_prefix=False):
    """The captured bucket a lane-frame call of ``rounds`` rounds at ``padded_len`` replays, or None (eager): the
    shape must match a captured bucket exactly (the trace binds the frame's tensors), the call must be a single
    chunk from position 0 (a continuation carries a sliding tail no replay can run)."""
    if not captured or not single_chunk or cached_prefix:
        return None
    key = (int(rounds), int(padded_len))
    return key if key in captured else None


def lane_trace_rounds_for_call(captured, padded_len, max_rounds):
    """How many rounds a call at ``padded_len`` should take so it replays a trace: the largest captured ``k`` at
    that length not above ``max_rounds``, else ``max_rounds`` (eager)."""
    ks = sorted(k for k, s in captured or () if s == int(padded_len) and k <= int(max_rounds))
    return ks[-1] if ks else int(max_rounds)


def fit_table_width(table, width):
    """A page table trimmed or null-padded on its last dim to ``width`` columns (the width the trace captured, so
    the persistent buffer never grows under a live trace); None and non-tensors pass through."""
    if table is None or not hasattr(table, "shape"):
        return table
    import torch

    width = int(width)
    cur = int(table.shape[-1])
    if cur == width:
        return table
    if cur > width:
        return table[..., :width]
    pad = torch.zeros(*table.shape[:-1], width - cur, dtype=table.dtype)
    return torch.cat([table, pad], dim=-1)
