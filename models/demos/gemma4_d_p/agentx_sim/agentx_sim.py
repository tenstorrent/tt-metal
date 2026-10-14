#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replay AgentX traces through the Gemma4 continuous-batching cost model (host only).

    agentx_sim.py --traces .../cc-traces-weka-062126-256k/traces.jsonl \
        --config 8192x8192:fcfs --config 2048x8192:pack --concurrency 16,32,64 [--cache inf|<tokens>]

Workload (closed loop, like AIPerf's AgentX replay):
- Each session is a DAG. Main-agent requests chain in order; each sub-agent chain starts after the main
  request just before it, and a main request waits for the sub-agents that finished before it was sent.
- A request is sent when its dependencies have their responses (prefill done + out / decode_tps), plus the
  recorded gap since the latest dependency's recorded end (t + api_time), capped at --max-think-s.
- --concurrency sessions run at once; when one finishes the next starts.

Prefill:
- Prefix cache keyed by (session, hash_id), 64-token blocks: infinite or LRU over a token capacity. Lookup at
  arrival, insert when the request's prefill finishes.
- The request resumes at its cached depth rounded down to the chunk unit (--align chunk, today's block-cyclic
  layout) or exactly (--align exact). Its rows are padded up to whole units.
- Step cost is Model.step() from cost_model.py: dense(M) + per-request attention at its own depth.
- Scheduler policies as in cost_model.simulate(): fcfs, srpt, pack. Unbatched = unit == budget. Under pack a request
  that takes k slots runs as k unit lanes (--pieces lanes, the 2k-lanes runtime) or one k-unit chunk (wide).

TTFT is from send to prefill done (the decode's first token is not modelled).
"""

import argparse
import array
import collections
import heapq
import itertools
import json
import math
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_model import Model  # noqa: E402

BLOCK = 64
BUCKETS = [(0, 1024, "<=1K"), (1024, 4096, "1-4K"), (4096, 16384, "4-16K"), (16384, 1 << 40, ">16K")]


# ---- workload ----
def load_sessions(path):
    """Parse traces.jsonl into sessions of nodes; cached as a pickle next to the traces."""
    cache = path + ".sim.pkl"
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        with open(cache, "rb") as f:
            return pickle.load(f)
    sessions = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            assert d["block_size"] == BLOCK and d["hash_id_scope"] == "local"
            nodes = []  # dict(in, out, t, end, hashes, deps)

            def add(r, deps):
                nodes.append(
                    {
                        "in": r["in"],
                        "out": r["out"],
                        "t": r["t"],
                        "end": r["t"] + r.get("api_time", 0.0),
                        "hashes": array.array("l", r["hash_ids"]),
                        "deps": deps,
                    }
                )
                return len(nodes) - 1

            reqs = sorted(d["requests"], key=lambda r: r["t"])
            mains = [r for r in reqs if r["type"] != "subagent"]
            subs = [r for r in reqs if r["type"] == "subagent" and r["requests"]]
            main_ids, prev = [], None
            # sub-agents launched after main i (index into mains, -1 = before the first main)
            by_trigger = collections.defaultdict(list)
            for g in subs:
                i = max((k for k, m in enumerate(mains) if m["t"] <= g["t"]), default=-1)
                by_trigger[i].append(g)
            pending_subs = []  # (recorded end, last node id)

            def launch(trigger_id, groups):
                for g in groups:
                    last = trigger_id
                    for r in sorted(g["requests"], key=lambda r: r["t"]):
                        last = add(r, [last] if last is not None else [])
                    pending_subs.append((g["t"] + g.get("duration_ms", 0) / 1000.0, last))

            launch(None, by_trigger.get(-1, []))
            for i, m in enumerate(mains):
                deps = [prev] if prev is not None else []
                done = [s for s in pending_subs if s[0] <= m["t"] + 1.0]
                pending_subs[:] = [s for s in pending_subs if s[0] > m["t"] + 1.0]
                deps += [nid for _, nid in done]
                prev = add(m, deps)
                main_ids.append(prev)
                launch(prev, by_trigger.get(i, []))
            sessions.append(nodes)
    with open(cache, "wb") as f:
        pickle.dump(sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
    return sessions


class PrefixCache:
    """Blocks keyed by (session << 32) | hash_id. capacity None = infinite, else LRU in tokens."""

    def __init__(self, capacity_tokens=None):
        self.cap = None if capacity_tokens is None else capacity_tokens // BLOCK
        self.s = set() if self.cap is None else collections.OrderedDict()

    def lookup(self, sess, hashes):
        base, s, k = sess << 32, self.s, 0
        for h in hashes:
            if base | h not in s:
                break
            k += 1
        return k * BLOCK

    def insert(self, sess, hashes):
        base = sess << 32
        if self.cap is None:
            self.s.update(base | h for h in hashes)
            return
        s = self.s
        for h in reversed(hashes):  # the head ends up most recent, so eviction takes prefix tails first
            key = base | h
            if key in s:
                s.move_to_end(key)
            else:
                s[key] = None
        while len(s) > self.cap:
            s.popitem(last=False)


# ---- simulation ----
def run(
    model,
    sessions,
    C,
    budget,
    policy,
    concurrency,
    cache_tokens=None,
    align="chunk",
    max_think=60.0,
    decode_tps=100.0,
    hit_load_us_per_ktok=0.0,
    mode="pess",
    stagger_s=10.0,
    seed=0,
    max_sessions=None,
    pieces="lanes",
    host_ms=0.0,
    scale=1.0,
):
    rng = random.Random(seed)
    order = list(range(len(sessions)))[:max_sessions]
    slots = budget // C
    cache = PrefixCache(cache_tokens)
    ideal = PrefixCache(None) if cache_tokens is not None else None  # what an unlimited cache would hit

    # flat node state, filled as sessions start
    N = sum(len(sessions[s]) for s in order)
    sess_of, local_of = [0] * N, [0] * N
    nleft = [0] * N
    ready = [0.0] * N  # latest dependency's response time
    rec_ready = [0.0] * N  # latest dependency's recorded end
    children = [[] for _ in range(N)]
    base_of = {}
    stats = []  # per request: (send, done, in, cached, start, rows)
    arrivals = []  # heap (send time, node)
    left_in_session, sess_done_t = {}, {}
    next_sess = 0
    nxt = 0
    t_fed = [None]  # when the last session started: the load is at full concurrency until then

    def start_session(t):
        nonlocal next_sess, nxt
        if next_sess >= len(order):
            return
        s = order[next_sess]
        next_sess += 1
        if next_sess == len(order):
            t_fed[0] = t
        nodes = sessions[s]
        base_of[s] = nxt
        for i, n in enumerate(nodes):
            g = nxt + i
            sess_of[g], local_of[g] = s, i
            nleft[g] = len(n["deps"])
            for dpd in n["deps"]:
                children[nxt + dpd].append(g)
        left_in_session[s] = len(nodes)
        sess_done_t[s] = t
        for i, n in enumerate(nodes):
            if not n["deps"]:
                heapq.heappush(arrivals, (t, nxt + i))
        nxt += len(nodes)

    for _ in range(min(concurrency, len(order))):
        start_session(rng.uniform(0.0, stagger_s))

    t = 0.0
    busy = 0.0
    queue_fcfs = collections.deque()
    queue_rem = []  # heap (remaining rows, seq, node)
    seq = 0
    req = {}  # node -> [send, in, cached, start, need, progress]
    rows_total = 0
    steps = 0
    step_log = []  # (t_end, dt, rows)

    def admit(g, send):
        nonlocal seq
        n = sessions[sess_of[g]][local_of[g]]
        cached = cache.lookup(sess_of[g], n["hashes"])
        ideal_hit = ideal.lookup(sess_of[g], n["hashes"]) if ideal else cached
        cap = n["in"] - 1  # at least one token is computed
        start = min(cached, cap)
        if align == "chunk":
            start = (start // C) * C
        need = math.ceil((n["in"] - start) / C) * C
        req[g] = [send, n["in"], cached, start, need, 0, True, ideal_hit]
        if policy == "fcfs":
            queue_fcfs.append(g)
        else:
            heapq.heappush(queue_rem, (need, seq, g))
            seq += 1

    def finish(g, t_done):
        send, n_in, cached, start, need, _, _, ideal_hit = req.pop(g)
        s, i = sess_of[g], local_of[g]
        n = sessions[s][i]
        stats.append((send, t_done, n_in, cached, start, need, ideal_hit))
        cache.insert(s, n["hashes"])
        if ideal:
            ideal.insert(s, n["hashes"])
        resp = t_done + n["out"] / decode_tps
        sess_done_t[s] = max(sess_done_t[s], resp)
        for c in children[g]:
            ready[c] = max(ready[c], resp)
            rec_ready[c] = max(rec_ready[c], n["end"])
            nleft[c] -= 1
            if nleft[c] == 0:
                cn = sessions[s][local_of[c]]
                think = min(max(cn["t"] - rec_ready[c], 0.0), max_think)
                heapq.heappush(arrivals, (ready[c] + think, c))
        left_in_session[s] -= 1
        if left_in_session[s] == 0:
            start_session(sess_done_t[s])

    while arrivals or req:
        while arrivals and arrivals[0][0] <= t:
            send, g = heapq.heappop(arrivals)
            admit(g, send)
        if not req:
            t = arrivals[0][0]
            continue
        # pick this step's pieces: [(node, units)]
        run_list = []
        if policy == "fcfs":
            # admission order; a request keeps its slot until done (unbatched: one request, one chunk per step)
            for g in itertools.islice(queue_fcfs, slots):
                run_list.append((g, 1))
        else:
            free = slots
            while queue_rem and free > 0:
                rem, _, g = heapq.heappop(queue_rem)
                take = min(free, rem // C) if policy == "pack" else 1
                run_list.append((g, take))
                free -= take
        batch = []
        extra = 0.0
        for g, k in run_list:
            r = req[g]
            if pieces == "lanes":  # k unit-wide lanes of one request (the 2k-lanes runtime)
                batch += [(C, r[3] + r[5] + j * C) for j in range(k)]
            else:  # one k-unit-wide chunk
                batch.append((k * C, r[3] + r[5]))
            if r[6]:  # first piece: load the cached prefix
                extra += hit_load_us_per_ktok * r[3] / 1000.0
                r[6] = False
        dt = (model.step(batch, mode)["total"] * scale + extra) / 1e6 + host_ms / 1e3
        t += dt
        busy += dt
        steps += 1
        step_log.append((t, dt, sum(k for _, k in run_list) * C))
        for g, k in run_list:
            r = req[g]
            r[5] += k * C
            rows_total += k * C
            if r[5] >= r[4]:
                if policy == "fcfs":
                    queue_fcfs.remove(g)
                finish(g, t)
            elif policy != "fcfs":
                heapq.heappush(queue_rem, (r[4] - r[5], seq, g))
                seq += 1
    out = summarize(stats, t, busy, steps, rows_total)
    out["_stats"] = stats  # per request (send, done, in, cached, start, rows); not written to --json
    # Steady state: from the end of a short ramp to when the last session started (full concurrency until then).
    hi = t_fed[0] or 0.0
    lo = min(600.0, 0.1 * hi)
    if hi - lo > 60.0:
        w_busy = sum(dt for te, dt, _ in step_log if lo < te <= hi)
        w_rows = sum(rw for te, _, rw in step_log if lo < te <= hi)
        w = [x for x in stats if lo < x[1] <= hi]
        w_useful = sum(x[2] - x[3] for x in w)
        w_useful_ideal = sum(x[2] - x[6] for x in w)
        ttft = [x[1] - x[0] for x in w]
        out.update(
            {
                "ss_window_s": (lo, hi),
                "ss_util": w_busy / (hi - lo),
                "ss_tok_s": w_rows / (hi - lo),
                "ss_useful_tok_s": w_useful / (hi - lo),
                "ss_useful_ideal_tok_s": w_useful_ideal / (hi - lo),
                "ss_ttft_p50": pct(ttft, 50),
                "ss_ttft_p99": pct(ttft, 99),
                "ss_requests": len(w),
            }
        )
    return out


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def summarize(stats, makespan, busy, steps, rows):
    ttft = [d - s for s, d, *_ in stats]
    n_in = sum(x[2] for x in stats)
    cached = sum(x[3] for x in stats)
    started = sum(x[4] for x in stats)
    useful = n_in - cached  # new to this cache
    ideal_hit = sum(x[6] for x in stats)
    useful_ideal = n_in - ideal_hit  # new to an unlimited cache: what must be prefilled at all
    out = {
        "requests": len(stats),
        "makespan_s": makespan,
        "busy_s": busy,
        "steps": steps,
        "ttft_p50": pct(ttft, 50),
        "ttft_p90": pct(ttft, 90),
        "ttft_p99": pct(ttft, 99),
        "rows": rows,
        "useful": useful,
        "useful_ideal": useful_ideal,
        "useful_ideal_tok_s_busy": useful_ideal / busy,
        "evict_recompute_pct": 100.0 * (ideal_hit - cached) / rows,
        "hit_rate_ideal": ideal_hit / n_in,
        "tok_s_busy": rows / busy,
        "useful_tok_s_busy": useful / busy,
        "tok_s_wall": rows / makespan,
        "useful_tok_s_wall": useful / makespan,
        "pad_pct": 100.0 * (rows - (n_in - started)) / rows,
        "recompute_pct": 100.0 * (cached - started) / rows,
        "hit_rate": cached / n_in,
        "hit_rate_used": started / n_in,
        "util": busy / makespan,
    }
    for lo, hi, name in BUCKETS:
        b = [d - s for s, d, i, c, *_ in stats if lo < i - c <= hi or (lo == 0 and i - c <= hi)]
        out[f"n_{name}"] = len(b)
        out[f"p50_{name}"], out[f"p99_{name}"] = pct(b, 50), pct(b, 99)
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--data", default=os.path.join(here, "model_data_revc.json"))
    ap.add_argument("--config", action="append", help="UNITxBUDGET:POLICY, e.g. 8192x8192:fcfs, 2048x8192:pack")
    ap.add_argument("--concurrency", default="32", help="comma-separated sessions in flight")
    ap.add_argument("--cache", default="inf", help="inf, or LRU capacity in tokens (e.g. 20e6)")
    ap.add_argument("--align", default="chunk", choices=["chunk", "exact"])
    ap.add_argument("--max-think-s", type=float, default=60.0)
    ap.add_argument("--decode-tps", type=float, default=100.0, help="per-request decode rate for the closed loop")
    ap.add_argument("--hit-load-us-per-ktok", type=float, default=0.0, help="cost to load a cached prefix")
    ap.add_argument("--mode", default="pess", choices=["pess", "opt"])
    ap.add_argument(
        "--host-ms", type=float, default=0.0, help="host overhead per step (silicon replay: 3.3 serial, 6.3 batched)"
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply device step time (silicon replay: 8k 1.125, 4k 1.057, 2k 1.009, batched 1.054)",
    )
    ap.add_argument(
        "--pieces",
        default="lanes",
        choices=["lanes", "wide"],
        help="a request taking k slots runs as k unit lanes (the 2k-lanes runtime) or one k-unit chunk",
    )
    ap.add_argument("--sessions", type=int, default=None, help="first N sessions only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="append one JSON line per run here")
    a = ap.parse_args()

    model = Model(json.load(open(a.data)))
    sessions = load_sessions(a.traces)
    cache_tokens = None if a.cache == "inf" else int(float(a.cache))
    configs = a.config or ["8192x8192:fcfs", "2048x8192:pack"]
    print(
        f"{len(sessions)} sessions, {sum(map(len, sessions))} requests; cache {a.cache}, align {a.align}, "
        f"think cap {a.max_think_s:g} s, decode {a.decode_tps:g} tok/s, attention {a.mode}"
    )
    hdr = (
        f"{'config':>16} {'conc':>4} | {'TTFT p50':>8} {'p90':>7} {'p99':>7} | "
        + " ".join(f"{n:>11}" for _, _, n in BUCKETS)
        + f" | {'tok/s':>6} {'useful':>6} {'pad%':>5} {'recmp%':>6} {'hit':>5} {'used':>5} {'util':>5} {'wall h':>6}"
        + f" | {'ss util':>7} {'ss tok/s':>8} {'ss useful':>9} {'ss p50':>6} {'ss p99':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cfg in configs:
        cb, policy = cfg.split(":")
        C, B = (int(x) for x in cb.split("x"))
        for conc in (int(x) for x in a.concurrency.split(",")):
            r = run(
                model,
                sessions,
                C,
                B,
                policy,
                conc,
                cache_tokens,
                a.align,
                a.max_think_s,
                a.decode_tps,
                a.hit_load_us_per_ktok,
                a.mode,
                seed=a.seed,
                max_sessions=a.sessions,
                pieces=a.pieces,
                host_ms=a.host_ms,
                scale=a.scale,
            )
            b = " ".join(f"{r[f'p50_{n}']:5.2f}/{r[f'p99_{n}']:5.1f}" for _, _, n in BUCKETS)
            print(
                f"{cfg:>16} {conc:>4} | {r['ttft_p50']:8.2f} {r['ttft_p90']:7.2f} {r['ttft_p99']:7.2f} | {b} | "
                f"{r['tok_s_busy'] / 1e3:5.1f}k {r['useful_tok_s_busy'] / 1e3:5.1f}k {r['pad_pct']:5.1f} "
                f"{r['recompute_pct']:6.1f} {r['hit_rate']:5.3f} {r['hit_rate_used']:5.3f} {r['util']:5.2f} "
                f"{r['makespan_s'] / 3600:6.2f} | "
                + (
                    f"{r['ss_util']:7.2f} {r['ss_tok_s'] / 1e3:7.1f}k {r['ss_useful_tok_s'] / 1e3:8.1f}k "
                    f"{r['ss_ttft_p50']:6.2f} {r['ss_ttft_p99']:6.1f}"
                    if "ss_util" in r
                    else "  (no steady window)"
                ),
                flush=True,
            )
            if a.json:
                with open(a.json, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "config": cfg,
                                "concurrency": conc,
                                "cache": a.cache,
                                "align": a.align,
                                "max_think_s": a.max_think_s,
                                "decode_tps": a.decode_tps,
                                "mode": a.mode,
                                "pieces": a.pieces,
                                **{k: v for k, v in r.items() if not k.startswith("_")},
                            }
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
