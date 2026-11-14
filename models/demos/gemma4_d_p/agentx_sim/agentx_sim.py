#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replay AgentX traces through the Gemma4 continuous-batching cost model (host only).

    agentx_sim.py --traces .../cc-traces-weka-062126-256k/traces.jsonl --config 8192x8192:fcfs,2048x8192:pack \
        --concurrency 16,32,64 [--cache inf,4e6] [--align chunk,resume,exact] [--mid-cost 0.015] [-j 16] \
        [--json runs.jsonl] [--ttft-out ttft.jsonl]

Every knob but --sessions and --seed takes a comma list; the runs are their product, -j at a time (the traces are
parsed once and shared with the forked workers). One row per run, and one JSON line per run with every metric.
--ttft-out writes each run's per-request TTFT and new tokens, for plot_ttft_cdf.py.

Workload (closed loop, like AIPerf's AgentX replay):
- Each session is a DAG. Main-agent requests chain in order; each sub-agent chain starts after the main
  request just before it, and a main request waits for the sub-agents that finished before it was sent.
- A request is sent when its dependencies have their responses (prefill done + out / decode_tps), plus the
  recorded gap since the latest dependency's recorded end (t + api_time), capped at --max-think-s.
- --concurrency sessions run at once; when one finishes the next starts.

Prefill:
- Prefix cache keyed by (session, hash_id), 64-token blocks: infinite or LRU over a token capacity. Lookup at
  arrival, insert when the request's prefill finishes.
- The request resumes at its cached depth rounded down to the chunk unit (--align chunk), to Gemma4's resume granularity
  (--align resume: a per-rank slab, unit / CP; Gemma4PrefillRuntime.resume_granularity), or exactly (--align exact).
  Its rows are padded up to whole units. A lane that starts off a unit boundary costs --mid-cost more (measured:
  +0.015 at 2048, +0.02 at 8192).
- Each step runs budget / unit slots (unbatched: unit == budget). fcfs: admission order, a request keeps its slot until
  done; pack: the requests with the fewest rows left first, each taking every slot it still needs. A request's k slots
  run as k unit-wide lanes (the 2k-lanes runtime).
- Step time is Model.step_s from cost_model.py (dense(M) + per-request attention at its own depth, calibrated on
  silicon; a config without its own fit is marked *), plus the scheduler's gap between steps, which advances the
  clock but is not busy time.

TTFT is from send to prefill done (the decode's first token is not modelled).
"""

import argparse
import array
import collections
import heapq
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_model import CP, DEFAULT_DATA, Model  # noqa: E402

BLOCK = 64
# One finished request: sent / done (s), input tokens, cached prefix, where it resumed, rows computed, and the cached
# prefix an unlimited cache would have had.
Stat = collections.namedtuple("Stat", "send done n_in cached start rows ideal_hit")
BUCKETS = [(-1, 1024, "<=1K"), (1024, 4096, "1-4K"), (4096, 16384, "4-16K"), (16384, 1 << 40, ">16K")]


# ---- workload ----
# The parse is cached next to the traces as raw arrays (numbers only, nothing is executed on load): ~1 s to load
# against ~8 s to parse. (field, array typecode); hash_off / dep_off / sess_off index the flat hashes / deps / nodes.
CACHE_FIELDS = (
    ("in", "q"),
    ("out", "q"),
    ("t", "d"),
    ("end", "d"),
    ("hashes", "l"),
    ("hash_off", "q"),
    ("deps", "q"),
    ("dep_off", "q"),
    ("sess_off", "q"),
)
# Part of the cache stamp: bump it when parse_sessions or CACHE_FIELDS change, so an old cache is parsed again.
CACHE_VERSION = 1


def load_sessions(path):
    """Sessions of nodes for traces.jsonl, from the cache in <path>.simcache/ when it matches the file."""
    cache = path + ".simcache"
    stamp = f"v{CACHE_VERSION} {os.path.getsize(path)} {os.path.getmtime(path)}"
    try:
        if _read_stamp(cache) == stamp:
            sessions = _read_cache(cache)
            if _read_stamp(cache) == stamp:  # not rewritten while it was read
                return sessions
    except OSError:
        pass
    sessions = parse_sessions(path)
    try:
        _write_cache(cache, sessions, stamp)
    except OSError:
        pass  # e.g. a read-only dataset directory: parse every run
    return sessions


def _read_stamp(cache):
    with open(os.path.join(cache, "source")) as f:
        return f.read()


def _replace_file(path, write):
    """Write a file under a temporary name and rename it into place, so a reader never sees it half written."""
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def _write_cache(cache, sessions, stamp):
    cols = {name: array.array(code) for name, code in CACHE_FIELDS}
    for col in ("hash_off", "dep_off", "sess_off"):
        cols[col].append(0)
    for nodes in sessions:
        for n in nodes:
            for name in ("in", "out", "t", "end"):
                cols[name].append(n[name])
            cols["hashes"].extend(n["hashes"])
            cols["hash_off"].append(len(cols["hashes"]))
            cols["deps"].extend(n["deps"])
            cols["dep_off"].append(len(cols["deps"]))
        cols["sess_off"].append(len(cols["in"]))
    os.makedirs(cache, exist_ok=True)
    source = os.path.join(cache, "source")
    # Invalidate first and stamp last, so a mix of old and new columns is never used.
    _replace_file(source, lambda f: None)
    for name, col in cols.items():
        _replace_file(os.path.join(cache, name + ".bin"), col.tofile)
    _replace_file(source, lambda f: f.write(stamp.encode()))


def _read_cache(cache):
    cols = {}
    for name, code in CACHE_FIELDS:
        col = array.array(code)
        file = os.path.join(cache, name + ".bin")
        with open(file, "rb") as f:
            col.fromfile(f, os.path.getsize(file) // col.itemsize)
        cols[name] = col
    hashes, hash_off, deps, dep_off = memoryview(cols["hashes"]), cols["hash_off"], cols["deps"], cols["dep_off"]
    sess_off = cols["sess_off"]
    return [
        [
            {
                "in": cols["in"][i],
                "out": cols["out"][i],
                "t": cols["t"][i],
                "end": cols["end"][i],
                "hashes": hashes[hash_off[i] : hash_off[i + 1]],
                "deps": list(deps[dep_off[i] : dep_off[i + 1]]),
            }
            for i in range(sess_off[k], sess_off[k + 1])
        ]
        for k in range(len(sess_off) - 1)
    ]


def parse_sessions(path):
    """Parse traces.jsonl into sessions of nodes."""
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
            prev = None
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
                # Only sub-agents that ended before this main request was sent fed it.
                done = [s for s in pending_subs if s[0] <= m["t"]]
                pending_subs[:] = [s for s in pending_subs if s[0] > m["t"]]
                deps += [nid for _, nid in done]
                prev = add(m, deps)
                launch(prev, by_trigger.get(i, []))
            sessions.append(nodes)
    return sessions


class PrefixCache:
    """64-token blocks per session. capacity None = infinite, else LRU in tokens.

    A session's hash ids are prefix-chained (a block id fixes its position and predecessor), and insert refreshes a
    request's blocks tail first (so does a lookup hit, as a server pins the blocks it reuses), so the cached blocks of a
    session always form a prefix closed set: lookup is a binary
    search for the longest cached prefix, and an infinite cache only adds the blocks past it."""

    def __init__(self, capacity_tokens=None):
        self.cap = None if capacity_tokens is None else capacity_tokens // BLOCK
        if self.cap is None:
            self.sessions = collections.defaultdict(set)
        else:
            self.lru = collections.OrderedDict()  # (session << 32) | hash_id

    def _cached_blocks(self, sess, hashes):
        if self.cap is None:
            s, base = self.sessions[sess], 0
        else:
            s, base = self.lru, sess << 32
        lo, hi = 0, len(hashes)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if base | hashes[mid - 1] in s:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def lookup(self, sess, hashes):
        count = self._cached_blocks(sess, hashes)
        if self.cap is not None:
            base = sess << 32
            for h in reversed(hashes[:count]):
                self.lru.move_to_end(base | h)
        return count * BLOCK

    def insert(self, sess, hashes):
        if self.cap is None:
            self.sessions[sess].update(hashes[self._cached_blocks(sess, hashes) :])
            return
        base, s = sess << 32, self.lru
        for h in reversed(hashes):  # the head ends up most recent, so eviction takes prefix tails first
            key = base | h
            if key in s:
                s.move_to_end(key)
            else:
                s[key] = None
        for _ in range(len(s) - self.cap):
            s.popitem(last=False)


# ---- simulation ----


def resume_step(C, align):
    """Token multiple a request resumes at: the unit (chunk), Gemma4's resume granularity (resume: a per-rank slab;
    Gemma4PrefillRuntime.resume_granularity, which tests/unit/test_prefill_runner.py checks this against), or any
    token (exact)."""
    return {"chunk": C, "resume": C // CP, "exact": 1}[align]


def run(
    model,
    sessions,
    C,
    budget,
    policy,
    concurrency,
    *,
    cache_tokens=None,
    align="chunk",
    mid_cost=0.0,
    max_think_s=60.0,
    decode_tps=100.0,
    seed=0,
    max_sessions=None,
    calibrated=True,
    max_seq_len=262144,
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
    stats = []  # Stat per finished request
    arrivals = []  # heap (send time, node)
    freed = []  # heap of times a session's last response ends, freeing its concurrency slot
    left_in_session, sess_done_t = {}, {}
    next_sess = 0
    nxt = 0
    t_fed = None  # when the last session started: the load is at full concurrency until then

    def start_session(t):
        nonlocal next_sess, nxt, t_fed
        if next_sess >= len(order):
            return
        s = order[next_sess]
        next_sess += 1
        if next_sess == len(order):
            t_fed = t
        nodes = sessions[s]
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
        start_session(rng.uniform(0.0, 10.0))  # staggered over the first 10 s

    t = 0.0
    busy = 0.0
    queue_fcfs = collections.deque()
    queue_rem = []  # pack: heap (remaining rows, seq, node)
    seq = 0
    req = {}  # node -> [Stat (done not yet known), rows computed so far]
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
        step = resume_step(C, align)
        start = start // step * step
        # The runtime rejects a padded chunk that runs past the slot: near the end, resume earlier.
        while start and start + math.ceil((n["in"] - start) / C) * C > max_seq_len:
            start -= step
        need = math.ceil((n["in"] - start) / C) * C
        req[g] = [Stat(send, None, n["in"], cached, start, need, ideal_hit), 0]
        if policy == "fcfs":
            queue_fcfs.append(g)
        else:
            heapq.heappush(queue_rem, (need, seq, g))
            seq += 1

    def finish(g, t_done):
        stats.append(req.pop(g)[0]._replace(done=t_done))
        s, i = sess_of[g], local_of[g]
        n = sessions[s][i]
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
                think = min(max(cn["t"] - rec_ready[c], 0.0), max_think_s)
                heapq.heappush(arrivals, (ready[c] + think, c))
        left_in_session[s] -= 1
        if left_in_session[s] == 0:
            heapq.heappush(freed, sess_done_t[s])  # the next session starts when this one's response ends

    def release(until):
        # Start sessions in the order their predecessors' responses end, not the order their prefills end.
        while freed and freed[0] <= until:
            start_session(heapq.heappop(freed))

    while arrivals or req or freed:
        release(t)
        while arrivals and arrivals[0][0] <= t:
            send, g = heapq.heappop(arrivals)
            admit(g, send)
        if not req:
            if not arrivals and (not freed or next_sess >= len(order)):
                break  # nothing left to run: the clock stays at the last step's end
            t = min(arrivals[0][0] if arrivals else math.inf, freed[0] if freed else math.inf)
            continue
        # pick this step's pieces: [(node, units)]
        if policy == "fcfs":
            # admission order; a request keeps its slot until done (unbatched: one request, one chunk per step)
            run_list = [(g, 1) for g in itertools.islice(queue_fcfs, slots)]
        else:
            run_list, free = [], slots
            while queue_rem and free > 0:
                rem, _, g = heapq.heappop(queue_rem)
                take = min(free, rem // C)
                run_list.append((g, take))
                free -= take
        # k unit-wide lanes per request
        batch = [(C, req[g][0].start + req[g][1] + j * C) for g, k in run_list for j in range(k)]
        dt, gap = model.step_s(batch, C, budget, calibrated)
        dt *= 1 + mid_cost * sum(depth % C != 0 for _, depth in batch) / len(batch)
        t += dt
        busy += dt
        steps += 1
        step_log.append((t, dt, sum(k for _, k in run_list) * C))
        # Requests sent during the step look up the cache as it was when they were sent, before this step's
        # completions are inserted; they still wait for the next step.
        release(t)
        while arrivals and arrivals[0][0] < t:
            send, g = heapq.heappop(arrivals)
            admit(g, send)
        for g, k in run_list:
            r = req[g]
            r[1] += k * C
            rows_total += k * C
            if r[1] >= r[0].rows:
                if policy == "fcfs":
                    queue_fcfs.remove(g)
                finish(g, t)
            elif policy == "pack":
                heapq.heappush(queue_rem, (r[0].rows - r[1], seq, g))
                seq += 1
        t += gap  # the scheduler's time between steps: wall time, not busy
    out = summarize(stats, t, busy, steps, rows_total)
    out["_stats"] = stats  # not written to --json
    # Steady state: from the end of a short ramp to when the last session started (full concurrency until then).
    hi = t_fed or 0.0
    lo = min(600.0, 0.1 * hi)
    if hi - lo > 60.0:
        w_busy = sum(dt for te, dt, _ in step_log if lo < te <= hi)
        w_rows = sum(rw for te, _, rw in step_log if lo < te <= hi)
        w = [x for x in stats if lo < x.done <= hi]
        w_useful = sum(x.n_in - x.cached for x in w)
        w_useful_ideal = sum(x.n_in - x.ideal_hit for x in w)
        ttft = [x.done - x.send for x in w]
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


def json_record(r):
    """r for --json: empty buckets' NaN percentiles become null, since strict JSON has no NaN."""
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in r.items()}


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def summarize(stats, makespan, busy, steps, rows):
    ttft = [x.done - x.send for x in stats]
    n_in = sum(x.n_in for x in stats)
    cached = sum(x.cached for x in stats)
    started = sum(x.start for x in stats)
    useful = n_in - cached  # new to this cache
    ideal_hit = sum(x.ideal_hit for x in stats)
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
        b = [x.done - x.send for x in stats if lo < x.n_in - x.cached <= hi]
        out[f"n_{name}"] = len(b)
        out[f"p50_{name}"], out[f"p90_{name}"], out[f"p99_{name}"] = pct(b, 50), pct(b, 90), pct(b, 99)
    return out


G = {}  # the model, the sessions and the fixed knobs, shared with the forked workers


def parse_config(cfg, model):
    """UNITxBUDGET:POLICY -> (unit, budget, policy); SystemExit naming what the model data covers otherwise."""
    units = sorted(model.g)
    max_budget = max(m for m, _ in model.dense_pts["global"]) * CP
    try:
        unit_budget, policy = cfg.split(":")
        C, budget = (int(x) for x in unit_budget.split("x"))
    except ValueError:
        raise SystemExit(f"bad --config {cfg!r}: want UNITxBUDGET:POLICY, e.g. 2048x8192:pack")
    if policy not in ("fcfs", "pack"):
        raise SystemExit(f"bad --config {cfg!r}: POLICY is fcfs or pack")
    if C not in units:
        raise SystemExit(f"bad --config {cfg!r}: UNIT must be a measured chunk size, one of {units}")
    if budget % C or not C <= budget <= max_budget:
        raise SystemExit(f"bad --config {cfg!r}: BUDGET must be a multiple of UNIT, at most {max_budget} tokens a step")
    return C, budget, policy


def ttft_record(knobs, stats):
    """A --ttft-out line: the run's knobs, then each request's TTFT (s) and new tokens (input not in the cache), in
    finish order."""
    return {
        **knobs,
        "ttft_s": [round(x.done - x.send, 4) for x in stats],
        "new_tokens": [x.n_in - x.cached for x in stats],
    }


def run_config(job):
    """One run of the grid, job = (UNITxBUDGET:POLICY, concurrency, cache, align, max_think_s, decode_tps); returns
    its JSON record (the knobs, the metrics and the run's wall time), and its --ttft-out record when that is set."""
    cfg, concurrency, cache, align, max_think_s, decode_tps = job
    t0 = time.time()
    C, budget, policy = parse_config(cfg, G["model"])  # checked in main() before the workers fork
    r = run(
        G["model"],
        G["sessions"],
        C,
        budget,
        policy,
        concurrency,
        cache_tokens=None if cache == "inf" else int(float(cache)),
        align=align,
        mid_cost=G["mid_cost"],
        max_think_s=max_think_s,
        decode_tps=decode_tps,
        seed=G["seed"],
        max_sessions=G["max_sessions"],
    )
    knobs = dict(
        config=cfg, concurrency=concurrency, cache=cache, align=align, max_think_s=max_think_s, mid_cost=G["mid_cost"]
    )
    record = {
        **knobs,
        "uncalibrated": G["model"].calibration(C, budget)[1],
        "decode_tps": decode_tps,
        **{k: v for k, v in r.items() if not k.startswith("_")},
        "wall_s": round(time.time() - t0, 2),
    }
    return record, ttft_record({**knobs, "decode_tps": decode_tps}, r["_stats"]) if G["ttft_out"] else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--config", default="8192x8192:fcfs,2048x8192:pack", help="UNITxBUDGET:POLICY, POLICY fcfs or pack")
    ap.add_argument("--concurrency", default="32", help="sessions in flight")
    ap.add_argument("--cache", default="inf", help="inf, or LRU capacity in tokens (e.g. 20e6)")
    ap.add_argument("--align", default="chunk", help="chunk, resume or exact")
    ap.add_argument("--mid-cost", type=float, default=0.0, help="extra step time of a lane that starts mid-chunk")
    ap.add_argument("--max-think-s", default="60")
    ap.add_argument("--decode-tps", default="100", help="per-request decode rate for the closed loop")
    ap.add_argument("--sessions", type=int, default=None, help="first N sessions only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-j", type=int, default=1, help="parallel runs")
    ap.add_argument("--json", help="append one JSON line per run here")
    ap.add_argument("--ttft-out", help="append one JSON line per run with every request's TTFT here (plot_ttft_cdf.py)")
    a = ap.parse_args()

    t0 = time.time()
    split = lambda s, f=str: [f(x) for x in s.split(",") if x]
    G.update(model=Model.load(a.data), mid_cost=a.mid_cost, seed=a.seed, max_sessions=a.sessions, ttft_out=a.ttft_out)
    for cfg in split(a.config):  # before the traces load and the workers fork
        parse_config(cfg, G["model"])
    for align in split(a.align):
        if align not in ("chunk", "resume", "exact"):
            ap.error(f"bad --align {align!r}: choose chunk, resume or exact")
    G["sessions"] = load_sessions(a.traces)
    jobs = list(
        itertools.product(
            split(a.config),
            split(a.concurrency, int),
            split(a.cache),
            split(a.align),
            split(a.max_think_s, float),
            split(a.decode_tps, float),
        )
    )
    n_req = sum(map(len, G["sessions"][: a.sessions]))
    print(f"{n_req} requests; {len(jobs)} runs on {a.j} workers; traces loaded in {time.time() - t0:.1f} s")
    hdr = (
        f"{'config':>17} {'conc':>4} {'cache':>5} {'align':>6} {'think':>5} | {'TTFT p50':>8} {'p90':>6} {'p99':>6} | "
        f"{'tok/s':>6} {'useful':>6} {'pad%':>5} {'recmp%':>6} {'hit':>5} {'used':>5} | "
        f"{'ss util':>7} {'ss useful':>9} {'ss p50':>6} {'ss p99':>6}"
    )
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    uncalibrated = {}
    with mp.get_context("fork").Pool(a.j) as pool:
        for r, ttft in pool.imap(run_config, jobs):
            ss = (
                f"{r['ss_util']:7.2f} {r['ss_useful_ideal_tok_s'] / 1e3:8.1f}k {r['ss_ttft_p50']:6.2f} {r['ss_ttft_p99']:6.1f}"
                if "ss_util" in r
                else "(no steady window)"
            )
            if r["uncalibrated"]:
                uncalibrated[r["config"]] = r["uncalibrated"]
            print(
                f"{r['config'] + ('*' if r['uncalibrated'] else ''):>17} {r['concurrency']:>4} {r['cache']:>5} {r['align']:>6} {r['max_think_s']:5g} | "
                f"{r['ttft_p50']:8.2f} {r['ttft_p90']:6.2f} {r['ttft_p99']:6.1f} | "
                f"{r['tok_s_busy'] / 1e3:5.1f}k {r['useful_tok_s_busy'] / 1e3:5.1f}k {r['pad_pct']:5.1f} "
                f"{r['recompute_pct']:6.1f} {r['hit_rate']:5.3f} {r['hit_rate_used']:5.3f} | {ss}",
                flush=True,
            )
            if a.json:
                with open(a.json, "a") as f:
                    f.write(json.dumps(json_record(r), allow_nan=False) + "\n")
            if ttft:
                with open(a.ttft_out, "a") as f:
                    f.write(json.dumps(ttft) + "\n")
    for cfg, why in uncalibrated.items():
        print(f"* {cfg}: not fitted on silicon, {why}")


if __name__ == "__main__":
    main()
