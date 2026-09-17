# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Show how --maxschedchunk spreads test variants across workers, and across runs.

Every xdist worker owns one Tensix and nothing resets that core between tests, so
what precedes a measurement on its core is what can leak into it. This replays
xdist 3.8.0's LoadScheduling over a real collected item list to show what each
setting actually does to those sequences.

The scheduler logic is transcribed from
``xdist/scheduler/load.py`` (3.8.0), so chunk sizes and contiguity are exact:

    items_per_node_min = max(2, len(pending) // num_nodes // 4)
    items_per_node_max = max(2, len(pending) // num_nodes // 2)
    if len(node_pending) < items_per_node_min:
        if duration >= 0.1 and len(node_pending) >= 2:
            return                       # long tests: wait rather than top up
        num_send = items_per_node_max - len(node_pending)
        _send_tests(node, min(num_send, max(2 - len(node_pending), maxschedchunk)))

and ``_send_tests`` takes ``pending[:num]`` — a contiguous prefix of what is left,
which is why a chunk is a contiguous slice of collection order.

What is exact here: chunk sizes, contiguity, and which items share a worker for a
given completion order. What is modelled: per-item duration, and therefore the
completion order itself. Perf items all run far longer than the 0.1s branch
threshold, so the model only has to get their *relative* spread roughly right for
the comparison to hold.

    perf_sched_sim.py shard1_items.txt --nodes 15 --chunks 10 2000
"""

import argparse
import collections
import heapq
import pathlib
import random
import statistics

LONG_TEST_SECONDS = 0.1  # xdist's own "this node is doing long tests" threshold


def load_items(path):
    return [
        line.strip()
        for line in pathlib.Path(path).read_text().splitlines()
        if "::" in line
    ]


def module_of(nodeid):
    return nodeid.split("::", 1)[0]


def simulate(items, num_nodes, maxschedchunk, durations):
    """Replay LoadScheduling. Returns the ordered item list each worker ran."""
    pending = list(range(len(items)))
    node_pending = [collections.deque() for _ in range(num_nodes)]
    ran = [[] for _ in range(num_nodes)]
    chunk_sizes = []
    clock = [0.0] * num_nodes
    events = []  # (finish_time, node)

    def send(node, num):
        num = min(num, len(pending))
        if num <= 0:
            return
        chunk = pending[:num]
        del pending[:num]
        node_pending[node].extend(chunk)
        chunk_sizes.append(num)

    def check_schedule(node, duration):
        if not pending:
            return
        per_min = max(2, len(pending) // num_nodes // 4)
        per_max = max(2, len(pending) // num_nodes // 2)
        have = len(node_pending[node])
        if have < per_min:
            if duration >= LONG_TEST_SECONDS and have >= 2:
                return
            cap = max(2 - have, maxschedchunk)
            send(node, min(per_max - have, cap))

    # Initial distribution: two per node, then top up.
    for node in range(num_nodes):
        send(node, 2)
    for node in range(num_nodes):
        check_schedule(node, 0.0)

    for node in range(num_nodes):
        if node_pending[node]:
            item = node_pending[node].popleft()
            ran[node].append(item)
            clock[node] += durations[item]
            heapq.heappush(events, (clock[node], node, item))

    while events:
        _, node, item = heapq.heappop(events)
        check_schedule(node, durations[item])
        if node_pending[node]:
            nxt = node_pending[node].popleft()
            ran[node].append(nxt)
            clock[node] += durations[nxt]
            heapq.heappush(events, (clock[node], node, nxt))

    assert sum(len(r) for r in ran) == len(items), "lost items in the replay"
    return ran, chunk_sizes


def predecessors(ran):
    """item -> the item that ran immediately before it on the same worker."""
    pred = {}
    for seq in ran:
        for before, after in zip(seq, seq[1:]):
            pred[after] = before
    return pred


def worker_of(ran):
    return {item: node for node, seq in enumerate(ran) for item in seq}


def describe(items, ran, chunk_sizes, label):
    mods = [module_of(i) for i in items]
    sizes = [len(seq) for seq in ran]
    distinct = [len({mods[i] for i in seq}) for seq in ran]
    transitions = [
        sum(1 for a, b in zip(seq, seq[1:]) if mods[a] != mods[b]) for seq in ran
    ]
    print(f"\n=== {label}")
    print(f"  chunks sent           {len(chunk_sizes)}")
    print(
        f"  chunk size  median {int(statistics.median(chunk_sizes)):>5}   max {max(chunk_sizes):>5}"
    )
    print(
        f"  items per worker  min {min(sizes):>5}  median {int(statistics.median(sizes)):>5}  max {max(sizes):>5}"
    )
    print(f"  distinct modules per worker  min {min(distinct)}  max {max(distinct)}")
    print(
        f"  module changes on a worker   total {sum(transitions):>5}  max {max(transitions)}"
    )
    cross = sum(1 for s in chunk_sizes for _ in ())  # placeholder, computed below
    return {"sizes": sizes, "transitions": transitions}


def compare_runs(items, num_nodes, chunk, base_durations, jitter, seed_a, seed_b):
    """Two runs of the same code: how much of the sequencing survives?"""

    def jittered(seed):
        rng = random.Random(seed)
        return [d * (1 + rng.uniform(-jitter, jitter)) for d in base_durations]

    ran_a, _ = simulate(items, num_nodes, chunk, jittered(seed_a))
    ran_b, _ = simulate(items, num_nodes, chunk, jittered(seed_b))
    wa, wb = worker_of(ran_a), worker_of(ran_b)
    pa, pb = predecessors(ran_a), predecessors(ran_b)

    same_worker = sum(1 for i in range(len(items)) if wa[i] == wb[i])
    same_pred = sum(1 for i in range(len(items)) if pa.get(i) == pb.get(i))
    return {
        "same_worker": same_worker,
        "same_pred": same_pred,
        "n": len(items),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", help="file of collected node ids, one per line")
    ap.add_argument("--nodes", type=int, default=15)
    ap.add_argument("--chunks", type=int, nargs="+", default=[10, 2000])
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.03,
        help="per-item duration jitter between the two simulated runs (default 3%%)",
    )
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    items = load_items(a.items)
    mods = [module_of(i) for i in items]
    print(f"{len(items)} items, {len(set(mods))} modules, {a.nodes} workers")

    # A plausible cost model: a variant's cost tracks its module's share of the
    # shard, which is what makes some modules long runs of similar work.
    rng = random.Random(a.seed)
    base = [rng.lognormvariate(0.0, 0.5) for _ in items]

    for chunk in a.chunks:
        ran, sizes = simulate(items, a.nodes, chunk, base)
        describe(items, ran, sizes, f"--maxschedchunk={chunk}")
        cmp = compare_runs(items, a.nodes, chunk, base, a.jitter, a.seed, a.seed + 1)
        pct_w = 100 * cmp["same_worker"] / cmp["n"]
        pct_p = 100 * cmp["same_pred"] / cmp["n"]
        print(f"  two runs, {int(a.jitter * 100)}% duration jitter:")
        print(
            f"    same worker as last run     {cmp['same_worker']:>6} / {cmp['n']}  ({pct_w:.1f}%)"
        )
        print(
            f"    same predecessor on it      {cmp['same_pred']:>6} / {cmp['n']}  ({pct_p:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
