# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure (no ttnn, no device) remainder split of a spec fast-prefill segment (QWEN36_PREFILL_MASKED_BIG).

After the n_big full 8192-row big replays, a remainder of r rows (0 < r < 8192) is served by one of:
  old   : k = r // 2048 chunk replays (2048 rows each) + a masked short-trace tail bucket (128..2048) for r % 2048
  m4096 : one masked 4096-row short trace (valid_len = r; needs r <= 4096)
  m8192 : one masked 8192-row short trace (valid_len = r)
Every replay re-reads the KV written before it, so its time is modelled as cost(M, depth) = A[M] + B[M] * depth with
depth = the absolute position where the replay starts. The cheapest plan wins."""

import json
import os

# seconds: {rows M: (A, B per depth token)}; measured-shape defaults, overridable by QWEN36_PREFILL_COST_TABLE.
DEFAULT_COST_TABLE = {
    2048: (0.238, 3.6e-6),
    4096: (0.42, 4.0e-6),
    8192: (0.78, 4.6e-6),
}
CHUNK = 2048
TAIL_BUCKETS = (128, 256, 512, 1024, 2048)
MASKED_BIG_BUCKETS = (4096, 8192)


def cost_table():
    """DEFAULT_COST_TABLE overridden per key by the JSON env QWEN36_PREFILL_COST_TABLE ({"2048": [A, B], ...})."""
    table = dict(DEFAULT_COST_TABLE)
    raw = os.environ.get("QWEN36_PREFILL_COST_TABLE", "").strip()
    if raw:
        for k, v in json.loads(raw).items():
            table[int(k)] = (float(v[0]), float(v[1]))
    return table


def tail_bucket(length):
    """Smallest masked tail bucket (128..2048) holding `length` rows."""
    for b in TAIL_BUCKETS:
        if length <= b:
            return b
    raise ValueError(f"tail {length} > {TAIL_BUCKETS[-1]}")


def replay_cost(table, rows, depth, bucket=None):
    """Seconds of one replay of `rows` (a table key) starting at `depth`. A tail bucket <= 2048 scales the 2048 entry's
    A linearly with bucket / 2048 and keeps its B."""
    if bucket is not None and bucket < CHUNK:
        a, b = table[CHUNK]
        return a * bucket / CHUNK + b * depth
    a, b = table[rows]
    return a + b * depth


def plan_costs(r, depth, table=None, buckets=MASKED_BIG_BUCKETS):
    """{plan name: cost} of every feasible plan for a remainder of r rows starting at `depth`.
    `buckets`: the masked big buckets that exist (4096 / 8192)."""
    table = cost_table() if table is None else table
    k, t = divmod(r, CHUNK)
    old = sum(replay_cost(table, CHUNK, depth + j * CHUNK) for j in range(k))
    if t:
        old += replay_cost(table, CHUNK, depth + k * CHUNK, bucket=tail_bucket(t))
    out = {"old": old}
    if r >= 3:  # a masked replay carries GDN state: its conv history needs 3 real rows
        if 4096 in buckets and r <= 4096:
            out["m4096"] = replay_cost(table, 4096, depth)
        if 8192 in buckets and r <= 8192:
            out["m8192"] = replay_cost(table, 8192, depth)
    return out


def plan_remainder(r, depth, enabled, table=None, buckets=MASKED_BIG_BUCKETS):
    """(plan, costs) for the remainder r (0 <= r < 8192 rows after the full big blocks) starting at absolute `depth`.
    plan is "old", "m4096" or "m8192"; enabled False (flag off) or r == 0 always gives "old" (costs {} when off)."""
    if not enabled:
        return "old", {}
    if r <= 0:
        return "old", {"old": 0.0}
    costs = plan_costs(r, depth, table, buckets)
    best = min(costs, key=lambda p: (costs[p], p != "old"))  # ties keep the existing plan
    return best, costs


def old_split(r):
    """(k chunk replays, tail rows) of the existing plan."""
    return divmod(r, CHUNK)
