# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The lane engine's prefill, host-only leaf: the knobs of the lane-parallel prefill, the decode frame's host shapes,
and the traced lane frames -- which ``kxS`` shapes are captured at boot and how a call's rounds replay them.

A lane-frame prefill call is ``k`` rounds of one prompt per lane, each column prefilling its ``k`` users as one
stream of ``k x S`` tokens (``Gemma4Generator.prefill_forward_lanes``). The bucket ``kxS`` names one captured
device trace per mesh (GEMMA4_LANE_PREFILL_TRACE_BUCKETS, default ``1x128``: the served c1 frame, one real lane and
scratch lanes); a step's rounds split over the captured frames by the replay cost model (``lane_prefill_trace_plan``),
every other call stays eager. Leaf module: host-only arithmetic, importable under the unit conftest's stubs
(``generator_vllm`` imports from ``generator`` at module scope and the stub of that module carries a fixed name
list, so every knob and shape rule the bridge needs lives here, not there).
"""

import os

LANE_PREFILL_TRACE_ENV = "GEMMA4_LANE_PREFILL_TRACE"
LANE_PREFILL_TRACE_BUCKETS_ENV = "GEMMA4_LANE_PREFILL_TRACE_BUCKETS"
LANE_LMHEAD_IN_TRACE_ENV = "GEMMA4_LANE_LMHEAD_IN_TRACE"
DEFAULT_BUCKETS = "1x128"
TRACE_SEQ_LENS = (128, 512, 1024, 2048, 4096)


def lane_prefill_trace_enabled():
    """GEMMA4_LANE_PREFILL_TRACE=1 (opt-in until the Galaxy window validates it): capture the lane-frame buckets at
    boot and replay them; off, every lane-frame call is eager."""
    return os.environ.get(LANE_PREFILL_TRACE_ENV, "0").lower() in ("1", "true", "yes")


TRACE_SHORT_PREFILL_MAX_ENV = "GEMMA4_TRACE_SHORT_PREFILL_MAX"  # generator_trace.trace_short_prefill_max's knob


def bounded_sliding_trace_max(short_max=None):
    """The longest frame a bounded-sliding mesh can capture: ``generator_trace.trace_short_prefill_max()`` (its
    environment knob read here, the leaf cannot import that module). Above it the prefill stashes the cross-chunk
    sliding tail with a host write, which a capture cannot record (TT_FATAL inside begin_trace_capture)."""
    if short_max is not None:
        return int(short_max)
    try:
        return int(os.environ.get(TRACE_SHORT_PREFILL_MAX_ENV, "128"))
    except ValueError:
        return 128


def parse_lane_prefill_trace_buckets(raw, *, max_chunk, lane_slots, bounded_sliding, short_max=None):
    """``{(k, S), ...}`` from ``"1x128,2x128"``: k rounds per column times the padded length S. A bucket needs
    ``k * S`` within the prefill chunk, ``k`` within the lane's slots, ``S`` a trace ISL, and under bounded sliding
    ``k == 1`` (the kernel-capped ring fill carries ONE valid length per device: a round's single prompt length) and
    ``S`` within ``bounded_sliding_trace_max`` (a longer frame records the sliding-tail stash's host write inside
    the capture). Malformed or out-of-range entries are dropped (returned separately for the log)."""
    buckets, dropped = set(), []
    s_max = bounded_sliding_trace_max(short_max) if bounded_sliding else None
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
            and (s_max is None or s <= s_max)
        )
        (buckets.add((k, s)) if ok else dropped.append(token))
    return buckets, dropped


def lane_prefill_trace_buckets(*, max_chunk, lane_slots, bounded_sliding, raw=None, short_max=None):
    """The configured buckets (empty when the trace is off)."""
    if not lane_prefill_trace_enabled():
        return set(), []
    raw = os.environ.get(LANE_PREFILL_TRACE_BUCKETS_ENV, DEFAULT_BUCKETS) if raw is None else raw
    return parse_lane_prefill_trace_buckets(
        raw, max_chunk=max_chunk, lane_slots=lane_slots, bounded_sliding=bounded_sliding, short_max=short_max
    )


def lane_kv_pool_blocks(model_kv):
    """Blocks of the smallest per-layer K pool (``kv[layer][0].shape[0]``): the range every layer's global rows may
    address. 0 when unknown (no cache, host stubs, anything that is not a list of per-layer (K, V) pairs)."""
    pools = []
    try:
        layers = list(model_kv or [])
    except TypeError:
        return 0
    for layer_kv in layers:
        try:
            pools.append(int(layer_kv[0].shape[0]))
        except (TypeError, IndexError, AttributeError, KeyError):
            continue
    return min(pools) if pools else 0


def lane_unique_block_rows(users, blocks, pool, width=None):
    """A ``[users, width]`` int32 page table whose first ``blocks`` columns give every user its own blocks,
    ``1 + i mod (pool - 1)`` (block 0, the null block, left alone), the rest null. Warmups and frame captures hand
    the batched paged fill one user per row: with every row on block 0 the concurrent writers collide, which wedged
    the batch-32 demo on P150x8 (indefinite stall after the first all_gather; the demo generator's ``_mock_tokens``
    override). ``pool`` <= 1 (unknown) keeps the null rows."""
    import torch

    users, blocks = int(users), int(blocks)
    width = blocks if width is None else max(int(width), blocks)
    table = torch.zeros(users, width, dtype=torch.int32)
    if int(pool) > 1 and users * blocks > 0:
        flat = 1 + torch.arange(users * blocks, dtype=torch.int32) % (int(pool) - 1)
        table[:, :blocks] = flat.reshape(users, blocks)
    return table


def lane_prefill_trace_choice(captured, rounds, padded_len, *, single_chunk=True, cached_prefix=False):
    """The captured bucket a lane-frame call of ``rounds`` rounds at ``padded_len`` replays, or None (eager): the
    shape must match a captured bucket exactly (the trace binds the frame's tensors), the call must be a single
    chunk from position 0 (a continuation carries a sliding tail no replay can run)."""
    if not captured or not single_chunk or cached_prefix:
        return None
    key = (int(rounds), int(padded_len))
    return key if key in captured else None


def lane_prefill_trace_bucket_for(captured, rounds, padded_len, *, single_chunk=True, cached_prefix=False):
    """The captured bucket a call of ``rounds`` rounds at ``padded_len`` replays after padding with scratch rounds:
    the exact match, else the SMALLEST captured ``k`` above ``rounds`` at that length (the pad rounds are all-scratch
    lanes: a replayed row each, no prompt); None when nothing captured covers the call (eager)."""
    exact = lane_prefill_trace_choice(
        captured, rounds, padded_len, single_chunk=single_chunk, cached_prefix=cached_prefix
    )
    if exact is not None:
        return exact
    if not captured or not single_chunk or cached_prefix:
        return None
    above = sorted(k for k, s in captured if s == int(padded_len) and k > int(rounds))
    return (above[0], int(padded_len)) if above else None


LANE_TRACE_CALL_MS_ENV = "GEMMA4_LANE_TRACE_CALL_MS"
LANE_TRACE_ROUND_MS_ENV = "GEMMA4_LANE_TRACE_ROUND_MS"
LANE_TRACE_FRAME_MS_ENV = "GEMMA4_LANE_TRACE_FRAME_MS"
LANE_TRACE_PLAN_SLACK_ENV = "GEMMA4_LANE_TRACE_PLAN_SLACK"
# A frame call's wall by frame size at 128 tokens per round: the Galaxy walls of window 13's stage 0 (2026-10-25,
# (8,4), the lm_head inside the trace, replay synchronized: 1x128 63 ms = 60 replay + 3 host, 8x128 323-333, 32x128
# ~1250; 16x128 interpolated at the 8-round rate). The per-round cost is NOT flat (41 ms on the 8-round frame, 39 on
# the 32-round one, 63 alone), which is what makes several 8-round calls earlier than one 32-round call.
DEFAULT_FRAME_MS = {1: 63.0, 8: 325.0, 16: 650.0, 32: 1250.0}
DEFAULT_CALL_MS = 40.0  # a frame size outside the table: this fixed cost ...
DEFAULT_ROUND_MS = 36.0  # ... plus this per round of 128 tokens (between the 8- and 32-round frames' rates)
DEFAULT_PLAN_SLACK = 0.10  # a plan may run this fraction longer in total than the fastest one to return prompts earlier


def lane_trace_costs(call_ms=None, round_ms=None):
    """The linear fallback's two terms (ms): per call and per 128-token round, from the arguments, else the
    environment (GEMMA4_LANE_TRACE_CALL_MS / GEMMA4_LANE_TRACE_ROUND_MS), else the defaults."""

    def _read(value, env, default):
        if value is not None:
            return float(value)
        try:
            return float(os.environ.get(env, default))
        except ValueError:
            return float(default)

    return _read(call_ms, LANE_TRACE_CALL_MS_ENV, DEFAULT_CALL_MS), _read(
        round_ms, LANE_TRACE_ROUND_MS_ENV, DEFAULT_ROUND_MS
    )


def parse_frame_ms(raw):
    """``{k: ms}`` from ``"1:75,8:295,32:1620"``; malformed entries are skipped."""
    table = {}
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            k, ms = token.split(":")
            table[int(k)] = float(ms)
        except ValueError:
            continue
    return table


def lane_trace_frame_costs(frames, padded_len, frame_ms=None, call_ms=None, round_ms=None):
    """``k -> ms`` of one call per captured frame size at ``padded_len``: the measured table (the argument, else
    GEMMA4_LANE_TRACE_FRAME_MS, else the Galaxy defaults), a size the table lacks from the linear fallback; the
    frame's work scales with ``padded_len / 128``."""
    table = dict(DEFAULT_FRAME_MS)
    table.update(parse_frame_ms(os.environ.get(LANE_TRACE_FRAME_MS_ENV, "")))
    if frame_ms:
        table.update({int(k): float(v) for k, v in dict(frame_ms).items()})
    per_call, per_round = lane_trace_costs(call_ms, round_ms)
    scale = int(padded_len) / 128.0
    return {int(k): float(table.get(int(k), per_call + int(k) * per_round)) * scale for k in frames}


def lane_trace_plan_slack(slack=None):
    if slack is not None:
        return max(0.0, float(slack))
    try:
        return max(0.0, float(os.environ.get(LANE_TRACE_PLAN_SLACK_ENV, DEFAULT_PLAN_SLACK)))
    except ValueError:
        return DEFAULT_PLAN_SLACK


def lane_prefill_trace_plan(
    captured,
    rounds,
    padded_len,
    *,
    single_chunk=True,
    cached_prefix=False,
    frame_ms=None,
    slack=None,
    call_ms=None,
    round_ms=None,
):
    """The calls a step's ``rounds`` run as over the captured frames at ``padded_len``: a list of ``(frame_k,
    real_rounds)`` in call order. None when no frame at this length is captured, or the call is not a single cold
    chunk (the eager walk).

    A frame computes all its rounds (a scratch round costs the same as a real one), so a call costs its frame's
    measured wall (``lane_trace_frame_costs``). Every multiset of frames that covers the rounds with a real round
    in every frame is a plan; its calls run cheapest-per-real-round first (the prompts of a cheap full frame
    return before a padded or dearer one), and the chosen plan has the smallest mean wait (the sum over real
    rounds of the time their call completes) among the plans whose total is within ``slack`` (default 10 %,
    GEMMA4_LANE_TRACE_PLAN_SLACK) of the fastest total; ties fall to fewer calls. Window 12 (one 32-round frame per
    step) made all 113 prompts of a 29-round step wait 1.6 s; on the window-13 walls (1: 63, 8: 325, 32: 1250 ms)
    the same step runs as three 8-round calls and five 1-round calls, 1.29 s in all, its prompts returning after
    0.33 / 0.65 / 0.98 / 1.04 .. 1.29 s (mean 0.74 s); window 13 served c128 TTFT 2463 -> 1692 ms."""
    if not single_chunk or cached_prefix or rounds <= 0:
        return None
    frames = sorted({int(k) for k, s in captured if int(s) == int(padded_len) and int(k) >= 1}, reverse=True)
    if not frames:
        return None
    cost = lane_trace_frame_costs(frames, padded_len, frame_ms, call_ms, round_ms)
    rounds = int(rounds)
    plans = []

    def enumerate_counts(i, left, chosen):
        if left <= 0:
            plans.append(list(chosen))
            return
        if i == len(frames):
            return
        f = frames[i]
        enumerate_counts(i + 1, left, chosen)  # none of this size
        c = 1
        while (c - 1) * f < left:  # the c-th frame still carries a real round
            enumerate_counts(i + 1, left - min(c * f, left), chosen + [(f, c)])
            c += 1

    enumerate_counts(0, rounds, [])
    best = None
    for plan in plans:
        calls, left = [], rounds
        for f, c in plan:  # larger frames take full loads first; the remainder pads the last frame of a size
            for _ in range(c):
                real = min(f, left)
                left -= real
                calls.append((f, real))
        if left != 0 or any(real <= 0 for _f, real in calls):
            continue
        calls.sort(key=lambda fr: (cost[fr[0]] / fr[1], -fr[0]))
        total, done, wait = 0.0, 0.0, 0.0
        for f, real in calls:
            total += cost[f]
            done += cost[f]
            wait += done * real
        best = best or []
        best.append((total, wait, len(calls), calls))
    if not best:
        return None
    fastest = min(t for t, _w, _n, _c in best)
    within = [b for b in best if b[0] <= fastest * (1.0 + lane_trace_plan_slack(slack)) + 1e-9]
    _total, _wait, _n, calls = min(within, key=lambda b: (b[1], b[2], b[0]))
    return list(calls)


def lane_trace_rounds_for_call(captured, padded_len, max_rounds):
    """How many rounds a call at ``padded_len`` should take so it replays a trace: the largest captured ``k`` at
    that length not above ``max_rounds``, else ``max_rounds`` (eager)."""
    ks = sorted(k for k, s in captured or () if s == int(padded_len) and k <= int(max_rounds))
    return ks[-1] if ks else int(max_rounds)


LANE_TRACE_READBACK_ROWS = (1, 2, 4, 8, 16, 32)


def lane_trace_readback_rows(n, rows=32):
    """The rows a replay reads back for ``n`` real rounds: the smallest of ``LANE_TRACE_READBACK_ROWS`` at or
    above ``n`` (capped at the frame's ``rows``). One row-slice program per size, all warmed at the first
    capture, so no call's real-round count compiles a program on the mesh (window 12: every first-seen count cost
    a program-cache miss of 0.6-0.8 s inside the call: the 860-1125 ms 8x128 calls)."""
    rows = int(rows)
    n = max(1, int(n))
    for r in LANE_TRACE_READBACK_ROWS:
        if r >= n:
            return min(r, rows)
    return rows


def lane_trace_gather_indices(real, last_r, seq_len, rows=32):
    """The ``rows`` row indices the lane trace's gather reads from the frame's ``[k * S, H]`` hidden states
    (the trace's third persistent input): real round ``r``'s last token at ``r * S + last_r[r]``, in the order of
    ``real``, then the first real round's row repeated for the padding rows (their logits are never read)."""
    real = [int(r) for r in real]
    if not real:
        raise ValueError("a lane frame call carries at least one real round")
    if len(real) > int(rows):
        raise ValueError(f"{len(real)} real rounds exceed the {rows}-row gather")
    idx = [r * int(seq_len) + int(last_r[r]) for r in real]
    return idx + [idx[0]] * (int(rows) - len(idx))


def lane_trace_capture_width(needed_blocks, max_seq_len, block_size):
    """Columns of the per-layer page tables a lane frame is captured with: the serving width (the blocks of
    ``max_seq_len``, what the plugin's tables carry), never below the frame's own ``needed_blocks``. A trace reads
    its page tables at the addresses it was captured with; a frame captured narrower would see its key's persistent
    buffers grow (reallocate) under the first wider eager install -- a prompt at an uncaptured length -- and every
    later replay would read the freed buffer."""
    needed = max(1, int(needed_blocks or 0))
    if not max_seq_len or not block_size:
        return needed
    return max(needed, -(-int(max_seq_len) // int(block_size)))


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


# ── the knobs of the lane-parallel prefill ─────────────────────────────────────────────────────────


def _env_flag(name, default):
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def lane_prefill_table_upload_enabled():
    """Every lane-parallel prefill call uploads its per-layer page tables into the persistent device buffers
    (GEMMA4_LANE_PREFILL_TABLE_UPLOAD, default 1). ``ttnn_prefill_forward`` resolves the stash through
    ``_page_tables_to_ttnn``, which hands back a key's existing buffers without refreshing them, so without the upload
    every call after a key's first prefills through that first call's block ids. 0 reproduces that pre-fix path (the
    before/after coherence evidence of the 2026-10-24 Galaxy window 6: 20/32 coherent at conc 32 against 32/32);
    never serve with it."""
    return _env_flag("GEMMA4_LANE_PREFILL_TABLE_UPLOAD", "1")


def lane_prefill_batched_enabled():
    """The rounds of a gathered step prefill as one lane-batched walk (GEMMA4_LANE_PREFILL_BATCHED, default 1): k
    users per lane as one batched stream per column. Off, every round walks on its own in the one-user-per-lane
    layout, which isolates the plugin's admission change from the model's batched walk on the device."""
    return _env_flag("GEMMA4_LANE_PREFILL_BATCHED", "1")


def lane_lmhead_gather_enabled():
    """A lane frame's lm_head runs once per call on the real rounds' last-token rows gathered on the device, and only
    those rows are read back (GEMMA4_LANE_LMHEAD_GATHER, default 1). Off, every round projects its own 32-row tile
    and reads the full tile from each lane: ~80 ms per round on the Galaxy (window 11: a 32-round frame 3.55 s of
    which ~2.6 s were the 32 readbacks; the 8-round 0.89 s, the 1-round 0.15 s), the term that made the padded
    frames lose to the eager walk."""
    return _env_flag("GEMMA4_LANE_LMHEAD_GATHER", "1")


def lane_lmhead_in_trace_enabled():
    """The lane frame's trace gathers the real rounds' last-token rows itself (``ttnn.embedding`` over the
    row-major hidden states with the ``[1, 32]`` index input) and projects them with lm_head inside the trace
    (GEMMA4_LANE_LMHEAD_IN_TRACE, default 1); a replay then reads a row slice of the trace's logits and runs no
    other program. Off, the trace returns the hidden states and the gathered lm_head runs eagerly after the replay
    (window 12: ~8 eager programs per call whose shapes follow the real-round count, so every first-seen count
    paid a program-cache miss inside the call)."""
    return _env_flag(LANE_LMHEAD_IN_TRACE_ENV, "1")


LANE_TRACE_SYNC_PROFILE_ENV = "GEMMA4_LANE_TRACE_SYNC_PROFILE"


def lane_trace_sync_profile_enabled():
    """Measurement only (GEMMA4_LANE_TRACE_SYNC_PROFILE=1): a replay synchronizes the mesh after ``execute_trace``
    so the perf phases split the device replay from the readback; never on in serving (the sync is a round trip)."""
    return _env_flag(LANE_TRACE_SYNC_PROFILE_ENV, "0")


# ── the decode frame's host shapes ─────────────────────────────────────────────────────────────────
# ``prepare_decode_inputs_host`` builds the step for the lane-major batch (``lanes x rows_per_lane`` rows): the hidden
# rows shard over the lane axis on dim 2, the RoPE positions (``[1, batch]`` uint32, the embedding lookup's layout) on
# dim 1, the cache positions (``[batch]`` int32: the paged cache update and the decode SDPA read their batch from
# dim 0) and the page table (``[batch, blocks]`` int32) on dim 0. A unit that hands the layer the frame must shard the
# same shapes: a 2-D ``[1, rows]`` tensor as the cache positions reads as ONE update index for a 32-row column
# ("Number of update_idxs (1) should match batch size (32)", window 5, 2026-10-24).


def lane_decode_frame_shapes(lanes, rows_per_lane, blocks_per_user, hidden):
    """``name -> (host shape, per-column shape, lane-shard dim)`` of the decode frame's tensors."""
    batch = int(lanes) * int(rows_per_lane)
    return {
        "hidden": ((1, 1, batch, int(hidden)), (1, 1, int(rows_per_lane), int(hidden)), 2),
        "rope_positions": ((1, batch), (1, int(rows_per_lane)), 1),
        "cache_positions": ((batch,), (int(rows_per_lane),), 0),
        "page_table": ((batch, int(blocks_per_user)), (int(rows_per_lane), int(blocks_per_user)), 0),
    }


def check_lane_decode_frame(lanes, rows_per_lane, **host_shapes):
    """Raise ValueError when a host tensor of the frame (``hidden``, ``rope_positions``, ``cache_positions``,
    ``page_table``: the shapes handed to ``from_torch``) would not shard to ``rows_per_lane`` rows per column, or
    has the wrong rank for the op that reads it. Returns the per-column shapes."""
    expected = lane_decode_frame_shapes(lanes, rows_per_lane, host_shapes.get("page_table", (0, 0))[-1], 0)
    out = {}
    for name, shape in host_shapes.items():
        if name not in expected:
            raise ValueError(f"unknown decode-frame tensor {name!r}")
        host, _column, dim = expected[name]
        shape = tuple(int(x) for x in shape)
        if len(shape) != len(host):
            raise ValueError(
                f"{name}: rank {len(shape)} {shape}, the frame needs rank {len(host)} "
                f"(the lane shard runs over dim {dim}; the paged cache update reads its batch from dim 0)"
            )
        if shape[dim] != host[dim]:
            raise ValueError(
                f"{name}: {shape[dim]} rows on the lane-shard dim {dim}, the frame has {host[dim]} "
                f"({lanes} lanes x {rows_per_lane} rows)"
            )
        column = list(shape)
        column[dim] = int(rows_per_lane)
        out[name] = tuple(column)
    rows = {name: shape[expected[name][2]] for name, shape in out.items()}
    if len(set(rows.values())) > 1:
        raise ValueError(f"the frame's tensors disagree on rows per column: {rows}")
    return out
