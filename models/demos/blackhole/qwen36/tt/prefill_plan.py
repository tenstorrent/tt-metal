# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure (no ttnn, no device) planning of a spec fast-prefill call: which trace replays run a prompt segment, which
replay holds each GDN prefix-cache save point, and the masked-big remainder choice.

Two flags use it (tp_common):
  * QWEN36_PREFILL_INTRACE_SNAP: a prefill call [s, Tp) runs as ONE segment; every save point p (s < p <= Tp - 3,
    128-aligned) is either the END of a replay (saved from the bound GDN state after it, as before) or INSIDE one
    (snapshotted by that replay: chunk_gated_delta_rule copies the carried state after the 32-row chunk ending at p,
    and the conv rows p-3..p-1 are gathered from the replay's conv input). Today's planner (spec_save_points ->
    segments [s, p1), [p1, p2), ..., [pk, Tp)) splits the prompt at every point, so a point inside an 8K chunk turns
    it into 2048-row chunk replays plus short-trace tails (an 8K final chunk at 24K: 1539 ms instead of 1174).
  * QWEN36_PREFILL_MASKED_BIG: the remainder after the full big blocks of a segment runs as one masked 4096 / 8192
    spec short trace when plan_remainder finds it cheaper than 2048-row chunk replays plus a short-trace tail.

Numerics. A replay computes the same per-row ops whatever the partition: the GDN state at a 32-row chunk boundary
depends only on the rows before it (the chunk op scans 32-row chunks in order), the paged SDPA reads the same K/V
and every matmul / norm is row-wise. Two things differ from a split run: (1) a split run passes the state at the
split through the end of a masked short-trace tail; any fully masked trailing 32-row chunk there is one more scan
step with beta = g = 0, which re-packs the state through the matrix engine (srcA/B hold the fp32 state at TF32)
instead of handing on the fp32 DST bytes, and (2) a masked big remainder has its own trailing masked chunks. Both
only touch the low mantissa bits of the carried GDN state at such boundaries; prefill_plan does not model them.
Measured on the TP8 row (2026-10-30): a snapshot saved after a masked tail differs from the in-replay one in the low
bits of every recurrent-state element (PCC 0.9999999), yet the prefill logits, the 64 greedy tokens, the feed row
and the final GDN state of 12 prompt lengths (100 .. 66536) and the logits of 36 resumed multi-turn prefills were bit
for bit those of the split run: the next scan step reads the carried state through the same TF32 operand path either
way. The masked big remainder is bit-identical where it replaces 2048-row chunks + a short-trace tail; where the split
run's tail is 1-2 rows (the eager masked bucket, e.g. a 2049-token prompt) the logits move (PCC 0.99978).

Cost model: one replay of a trace of M rows starting at absolute position `depth` costs A[M] + B[M] * depth seconds
(the paged SDPA re-reads the prefix), measured on the d21u14 TP8 row (r3/r5 stacks, production env); override per
key with the JSON env QWEN36_PREFILL_COST_TABLE ({"c2048": [A, B], ...})."""

import json
import os
from collections import namedtuple

CHUNK = 2048
BIG = 8192
SHORT_BUCKETS = (128, 256, 512, 1024, 2048)
MASKED_BIG_BUCKETS = (4096, 8192)
ALIGN = 128
CHUNK_ROWS = 32  # the GDN chunk op's chunk: in-replay snapshots sit on its boundaries

# seconds: key -> (A, B per depth token). c*: chunk-trace replays incl. the eager per-chunk drafter warm; s*: spec
# short traces (drafter warm in the trace); m*: masked big spec short traces (estimated from the chunk traces: the
# big chunk plus its drafter warm; 4096 = interpolated).
DEFAULT_COST_TABLE = {
    "c2048": (0.220, 4.3e-6),
    "c8192": (0.903, 7.8e-6),
    "s128": (0.056, 1.9e-6),
    "s256": (0.062, 4.0e-6),
    "s512": (0.078, 3.0e-6),
    "s1024": (0.121, 3.7e-6),
    "s2048": (0.235, 3.9e-6),
    "m4096": (0.460, 5.6e-6),
    "m8192": (0.903, 7.8e-6),
    # a tail of 1-2 rows (no short trace: the eager masked bucket, ~3K host-dispatched ops); conservative
    "eager": (0.250, 3.0e-6),
}

# One replay of a fast-prefill segment: kind "big" (big chunk trace, `rows` = its chunk), "c2k" (2048 chunk trace),
# "tail" (spec short trace of the smallest bucket >= rows, rows < 2048), "mbig" (masked big spec short trace of
# `bucket` rows serving `rows` real rows). start = absolute position of row 0.
Replay = namedtuple("Replay", "kind start rows bucket")


def cost_table():
    """DEFAULT_COST_TABLE overridden per key by QWEN36_PREFILL_COST_TABLE."""
    table = dict(DEFAULT_COST_TABLE)
    raw = os.environ.get("QWEN36_PREFILL_COST_TABLE", "").strip()
    if raw:
        for k, v in json.loads(raw).items():
            table[str(k)] = (float(v[0]), float(v[1]))
    return table


def tail_bucket(n):
    """Smallest short-trace bucket holding n rows (1 <= n < 2048 for a tail; 2048 itself too)."""
    for b in SHORT_BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"tail of {n} rows exceeds the {SHORT_BUCKETS[-1]} bucket")


def replay_cost(rep, table=None):
    table = cost_table() if table is None else table
    if rep.kind == "big":
        a, b = table[f"c{rep.rows}"] if f"c{rep.rows}" in table else table["c8192"]
    elif rep.kind == "c2k":
        a, b = table["c2048"]
    elif rep.kind == "tail":
        a, b = table["eager"] if rep.rows < 3 else table[f"s{rep.bucket}"]
    else:  # mbig
        a, b = table[f"m{rep.bucket}"]
    return a + b * rep.start


def plan_remainder(r, depth, buckets=MASKED_BIG_BUCKETS, table=None):
    """(plan, costs) for a remainder of r rows (0 < r < 8192, after the full big blocks) starting at absolute `depth`:
    plan "old" (r // 2048 chunk replays + a short-trace tail for r % 2048) or "m4096" / "m8192" (one masked spec short
    trace; needs 3 <= r <= bucket, the masked replay carries its conv history from its own rows). `buckets`: the masked
    big buckets captured. Ties keep "old"."""
    table = cost_table() if table is None else table
    if r <= 0:
        return "old", {"old": 0.0}
    k, t = divmod(r, CHUNK)
    old = sum(replay_cost(Replay("c2k", depth + j * CHUNK, CHUNK, CHUNK), table) for j in range(k))
    if t:
        old += replay_cost(Replay("tail", depth + k * CHUNK, t, tail_bucket(t)), table)
    costs = {"old": old}
    if r >= 3:
        for bk in MASKED_BIG_BUCKETS:
            if bk in buckets and r <= bk:
                costs[f"m{bk}"] = replay_cost(Replay("mbig", depth, r, bk), table)
    best = min(costs, key=lambda p: (costs[p], p != "old"))
    return best, costs


def segment_replays(a, b, big=BIG, masked_bucket=None, chunk=CHUNK):
    """The replays _fast_segment runs for the segment [a, b) (a % 128 == 0), in order. big: the big chunk trace's rows
    (0 = none); masked_bucket: the masked big bucket chosen for the remainder after the full big blocks (None = the
    2048 chunks + tail)."""
    n = b - a
    assert n > 0 and a % ALIGN == 0, (a, b)
    num_full, tail = divmod(n, chunk)
    if num_full == 0:
        return [Replay("tail", a, n, tail_bucket(n))]
    out = []
    if masked_bucket is not None:
        assert big, "a masked big remainder needs the big chunk trace"
        n_big = n // big
        out += [Replay("big", a + i * big, big, big) for i in range(n_big)]
        cs = a + n_big * big
        assert 3 <= b - cs <= masked_bucket, (a, b, masked_bucket)
        out.append(Replay("mbig", cs, b - cs, masked_bucket))
        return out
    rows = num_full * chunk
    n_big = rows // big if big else 0
    out += [Replay("big", a + i * big, big, big) for i in range(n_big)]
    pos = a + n_big * big
    out += [Replay("c2k", pos + j * chunk, chunk, chunk) for j in range((rows - n_big * big) // chunk)]
    if tail:
        out.append(Replay("tail", a + rows, tail, tail_bucket(tail)))
    return out


def assign_points(replays, points):
    """Map every save point to its replay: [(inside, end)] per replay, `inside` = the points strictly inside the
    replay's rows (start < p < start + rows, an in-replay snapshot), `end` = the point at its last row (p == start +
    rows, saved from the bound state after the replay) or None. Every point must fall in (first start, last end]."""
    out = [([], None) for _ in replays]
    for p in sorted(set(int(x) for x in points)):
        for i, rep in enumerate(replays):
            lo, hi = rep.start, rep.start + rep.rows
            if lo < p < hi:
                out[i][0].append(p)
                break
            if p == hi:
                out[i] = (out[i][0], p)
                break
        else:
            raise ValueError(f"save point {p} outside the replays {replays}")
    return out


def snap_index_words(rel_rows, slots, words=8, end_word=0x7FFFFFFF):
    """Device words of one replay's in-replay snapshots at row offsets `rel_rows` (sorted, 128-aligned, > 0):
    (rec, conv). rec: `words` uint32 for the chunk op's snapshot_index page, chunk r/32 - 1 per snapshot then end_word
    (>= any chunk count: ends the list). conv: 4 * slots row indices of the conv-input gather (snapshot k gathers rows
    r-3, r-2, r-1 into rows 3k..3k+2 of the gather output; unused entries read row 0). len(rel_rows) <= slots."""
    rel_rows = [int(r) for r in rel_rows]
    assert len(rel_rows) <= slots, f"{len(rel_rows)} in-replay snapshots exceed the {slots} slots"
    assert all(r >= ALIGN and r % CHUNK_ROWS == 0 for r in rel_rows), rel_rows
    assert rel_rows == sorted(set(rel_rows)), rel_rows
    rec = [r // CHUNK_ROWS - 1 for r in rel_rows] + [end_word] * (words - len(rel_rows))
    conv = []
    for r in rel_rows:
        conv += [r - 3, r - 2, r - 1]
    conv += [0] * (4 * slots - len(conv))
    return rec, conv


def plan_cost(a, b, points, one_segment, big=BIG, chunk=CHUNK, masked_choice=None, table=None):
    """Modelled seconds of a prefill call [a, b) with GDN save points `points` (a < p < b): one_segment=True runs it as
    one segment (in-replay snapshots), False splits it at every point (today's planner). masked_choice(n, start) -> the
    masked big bucket of a segment or None (QWEN36_PREFILL_MASKED_BIG; None = never). Saves cost the same either way
    and are left out; so is the in-replay staging (two 32 / 64-byte host writes per replay)."""
    table = cost_table() if table is None else table
    bounds = [a, b] if one_segment else [a] + sorted(p for p in set(points) if a < p < b) + [b]
    tot = 0.0
    for x, y in zip(bounds, bounds[1:]):
        mb = masked_choice(y - x, x) if (masked_choice is not None and y - x >= chunk) else None
        tot += sum(replay_cost(r, table) for r in segment_replays(x, y, big, mb, chunk))
    return tot
