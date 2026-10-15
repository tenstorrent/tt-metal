# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hit rate vs LRU cache capacity (tt-metal#60408, P0).

Usage: python3 -I agentx_lru_sweep.py <traces.jsonl> <concurrency> <capacity> [<capacity> ...]

capacity: tokens, with k/m suffixes (1024-based), e.g. 256k, 2m.
concurrency: sessions that share one cache at a time. 1 = each session has the
cache to itself. K > 1 = K sessions (file order) replay together from t=0,
sharing one cache. hash_ids are session-local, so sessions never share blocks,
they only compete for capacity.

Counting: leading run of cached blocks (prefix cache). A request's blocks are
touched tail-first so the first block is most recent and eviction removes the
tail of a prefix first.
"""
import json
import os
import sys
from collections import OrderedDict
from multiprocessing import Pool

BLOCK = 64
ID_BITS = 24  # block ids are offset by session index << ID_BITS to stay distinct


def parse_tokens(s):
    mult = {"k": 1024, "m": 1024 ** 2}
    return int(s[:-1]) * mult[s[-1]] if s[-1].lower() in mult else int(s)


def fmt_tokens(n):
    if n >= 1 << 40:
        return "infinite"
    return f"{n // 1024 ** 2}M" if n % 1024 ** 2 == 0 else f"{n // 1024}K"


def load_groups(path, k):
    """Yield lists of k sessions, each a list of its requests (sub-agents flattened)."""
    group = []
    with open(path) as f:
        for line in f:
            rows = []
            for r in json.loads(line)["requests"]:
                if r["type"] == "subagent":
                    rows.extend(r["requests"])
                else:
                    rows.append(r)
            group.append(rows)
            if len(group) == k:
                yield group
                group = []
    if group:
        yield group


def run(args):
    path, k, cap_tokens = args
    cap = cap_tokens // BLOCK
    total_in = new = 0
    for group in load_groups(path, k):
        events = [(r["t"], i, r) for i, rows in enumerate(group) for r in rows]
        events.sort(key=lambda e: e[:2])
        cache = OrderedDict()
        for _, i, r in events:
            ids = r["hash_ids"]
            if ids and max(ids) >= 1 << ID_BITS:
                raise ValueError("block id too large for session offset")
            keys = [h + (i << ID_BITS) for h in ids]
            lead = 0
            for key in keys:
                if key not in cache:
                    break
                lead += 1
            for key in reversed(keys):
                cache[key] = None
                cache.move_to_end(key)
            while len(cache) > cap:
                cache.popitem(last=False)
            total_in += r["in"]
            new += r["in"] - BLOCK * lead
    return cap_tokens, total_in, new


def main(path, k, caps):
    jobs = [(path, k, c) for c in caps] + [(path, k, 1 << 60)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        results = {c: (t, n) for c, t, n in pool.map(run, jobs)}
    inf_new = results[1 << 60][1]
    print(f"concurrency {k}; capacity is shared by {k} session(s)\n")
    print(f"{'capacity':>10s} {'hit rate':>9s} {'new tokens':>14s} {'extra vs infinite':>18s}")
    for c in sorted(results):
        total, new = results[c]
        print(f"{fmt_tokens(c):>10s} {100 * (1 - new / total):8.2f}% {new:14,d} {new - inf_new:18,d}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: python3 -I agentx_lru_sweep.py <traces.jsonl> <concurrency> <capacity> [...]")
    main(sys.argv[1], int(sys.argv[2]), [parse_tokens(a) for a in sys.argv[3:]])
