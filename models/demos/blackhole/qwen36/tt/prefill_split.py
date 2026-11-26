# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How a prefill segment runs the rows after its whole big blocks (Qwen36Model._prefill_traced_chunked_tp). Pure
Python, no ttnn.

The remainder of r rows (the full 2048-row chunks after the segment's whole 8192-row blocks, and its tail) runs by one
of these plans:

  "chunks": r // 2048 replays of the 2048-row chunk trace, then one masked-bucket forward (128 .. 2048 rows) of the
            r % 2048 rows left;
  "m4096":  one masked 4096-row forward of all r rows (r <= 4096);
  "m8192":  one masked 8192-row forward of all r rows (r <= 8192).

Every forward attends over the KV cache written before it, so its time is modelled as A[rows] + B[rows] * depth, depth
being the absolute position where it starts. The cheapest plan wins; a tie keeps "chunks"."""

# Seconds per forward of the 64-layer model at TP=8: {rows: (A, B per token of depth)}, measured on the production
# branch. A masked bucket of up to 2048 rows takes the 2048 entry's A scaled by bucket / 2048 and its B.
COST_TABLE = {
    2048: (0.238, 3.6e-6),
    4096: (0.42, 4.0e-6),
    8192: (0.78, 4.6e-6),
}
CHUNK = 2048
TAIL_BUCKETS = (128, 256, 512, 1024, 2048)
MASKED_BIG_BUCKETS = (4096, 8192)


def tail_bucket(rows):
    """The smallest masked bucket of up to 2048 rows that holds ``rows`` rows."""
    for bucket in TAIL_BUCKETS:
        if rows <= bucket:
            return bucket
    raise ValueError(f"a tail of {rows} rows is longer than the largest tail bucket, {TAIL_BUCKETS[-1]}")


def forward_cost(rows, depth, bucket=None, table=COST_TABLE):
    """Modelled seconds of one forward of ``rows`` rows (a COST_TABLE key) from ``depth``; ``bucket`` (up to 2048) scales
    the 2048 entry's A."""
    if bucket is not None and bucket < CHUNK:
        a, b = table[CHUNK]
        return a * bucket / CHUNK + b * depth
    a, b = table[rows]
    return a + b * depth


def plan_costs(r, depth, buckets=MASKED_BIG_BUCKETS, table=COST_TABLE):
    """{plan: modelled seconds} of every plan that can run a remainder of ``r`` rows from ``depth``; ``buckets`` are the
    masked big buckets that have a captured trace."""
    k, t = divmod(r, CHUNK)
    chunks = sum(forward_cost(CHUNK, depth + j * CHUNK, table=table) for j in range(k))
    if t:
        chunks += forward_cost(CHUNK, depth + k * CHUNK, bucket=tail_bucket(t), table=table)
    costs = {"chunks": chunks}
    for bucket in sorted(buckets):
        if r <= bucket:
            costs[f"m{bucket}"] = forward_cost(bucket, depth, table=table)
    return costs


def plan_remainder(r, depth, buckets=MASKED_BIG_BUCKETS, table=COST_TABLE):
    """(plan, costs) of a remainder of ``r`` rows (0 <= r <= 8192) from ``depth``: "chunks", "m4096" or "m8192"."""
    if r <= 0:
        return "chunks", {"chunks": 0.0}
    costs = plan_costs(r, depth, buckets, table)
    return min(costs, key=lambda p: (costs[p], p != "chunks")), costs
