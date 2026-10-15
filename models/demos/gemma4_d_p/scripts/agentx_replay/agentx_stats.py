# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AgentX workload stats (tt-metal#60408, P0).

Usage: python3 -I agentx_stats.py <path/to/traces.jsonl>

Cache model: infinite cache per session. hash_ids are session-local
(hash_id_scope == "local"), so blocks cannot be shared across sessions.
"""
import json
import sys

K = 1024  # bucket edges are 1K/4K/16K; the ticket doesn't say 1000 or 1024
BUCKETS = [("<=1K", 1 * K), ("1-4K", 4 * K), ("4-16K", 16 * K), (">16K", None)]
LIMIT = 256 * 1024  # Gemma4 max context
BLOCK = 64
EXPECTED_REQUESTS = 98_827
EXPECTED_INPUT_TOKENS = 21_635_381_376


def load_sessions(path):
    """Yield (session_id, rows) with sub-agent requests flattened in, sorted by t."""
    with open(path) as f:
        for line in f:
            sess = json.loads(line)
            rows = []
            for r in sess["requests"]:
                if r["type"] == "subagent":
                    rows.extend({**inner, "agent": "sub"} for inner in r["requests"])
                else:
                    rows.append({**r, "agent": "main"})
            rows.sort(key=lambda r: r["t"])
            yield sess["id"], rows


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, round(p / 100 * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def bucket(new_tokens):
    for name, hi in BUCKETS:
        if hi is None or new_tokens <= hi:
            return name


def main(path):
    reqs = []  # (agent, in, new_tokens)
    gaps = {"main": [], "all": []}
    for _, rows in load_sessions(path):
        seen = set()
        prev_t = {"main": None, "all": None}
        for r in rows:
            lead = 0
            for h in r["hash_ids"]:
                if h not in seen:
                    break
                lead += 1
            new_tokens = r["in"] - BLOCK * lead
            seen.update(r["hash_ids"])
            reqs.append((r["agent"], r["in"], new_tokens))
            for key in ("all", "main") if r["agent"] == "main" else ("all",):
                if prev_t[key] is not None:
                    gaps[key].append(r["t"] - prev_t[key])
                prev_t[key] = r["t"]

    n = len(reqs)
    total_in = sum(r[1] for r in reqs)
    total_new = sum(r[2] for r in reqs)
    print(f"requests {n:,} (expected {EXPECTED_REQUESTS:,}): "
          f"{'OK' if n == EXPECTED_REQUESTS else 'MISMATCH'}")
    print(f"input tokens {total_in:,} (expected {EXPECTED_INPUT_TOKENS:,}): "
          f"{'OK' if total_in == EXPECTED_INPUT_TOKENS else 'MISMATCH'}")

    print("\n== Cache (infinite, per session) ==")
    print(f"new tokens {total_new:,}; hit rate {100 * (1 - total_new / total_in):.2f}%")

    print("\n== Input length per request (tokens) ==")
    for label, sel in (("all", lambda r: True), ("main only", lambda r: r[0] == "main")):
        v = sorted(r[1] for r in reqs if sel(r))
        print(f"{label:10s} n={len(v):,} p50={pct(v, 50):,} p75={pct(v, 75):,} "
              f"p90={pct(v, 90):,} max={v[-1]:,}")

    print(f"\n== Share with input > {LIMIT:,} ==")
    for label, sel in (("all", lambda r: True), ("main only", lambda r: r[0] == "main")):
        sub = [r for r in reqs if sel(r)]
        over = sum(1 for r in sub if r[1] > LIMIT)
        print(f"{label:10s} {over:,} of {len(sub):,} requests ({100 * over / len(sub):.2f}%)")

    print(f"\n== New tokens per request, buckets (K={K}) ==")
    print(f"{'bucket':8s} {'requests':>10s} {'% reqs':>8s} {'new tokens':>16s} {'% new':>8s}")
    counts = {name: [0, 0] for name, _ in BUCKETS}
    for _, _, new in reqs:
        c = counts[bucket(new)]
        c[0] += 1
        c[1] += new
    for name, _ in BUCKETS:
        c, t = counts[name]
        print(f"{name:8s} {c:10,d} {100 * c / n:7.2f}% {t:16,d} {100 * t / total_new:7.2f}%")

    print("\n== Inter-arrival gaps within a session (seconds) ==")
    for key in ("main", "all"):
        v = sorted(gaps[key])
        print(f"{key:5s} n={len(v):,} p50={pct(v, 50):.2f} p90={pct(v, 90):.2f} "
              f"p99={pct(v, 99):.2f} max={v[-1]:.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
