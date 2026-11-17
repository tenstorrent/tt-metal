#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replay one AgentX run measured through tt-d-gen against the agentx_sim cost model (host only).

    dgen_replay.py <run dir> --traces .../cc-traces-weka-062126-256k/traces.jsonl [--cache slot,token,inf,engine] \
        [--policy rr,fcfs] [--inject-depth 9] [--request-overhead-ms 13.6] [--request-overhead-ms-per-1k 0.87] \
        [--chunk-time 153.8,1.081] [--json replay.json]

A run directory holds AIPerf's export (aiperf/profile_export.jsonl), the Dynamo frontend's log (frontend.log) and the
tt-d-gen prefill worker's log (prefill.log), as tt-metal#60869's runs do. Unbatched 8k chunks.

Measured, per AIPerf request: its send time and prompt length (the export), its prefill done (the frontend logs the
Prefill route span when it completes) and the engine request it became (the prefill worker's `prefill inject` lines:
slot, resume point, chunks). Each request is matched to its corpus entry for its block hashes.

Replayed: the same requests through cost_model.Model, one chunk per step, on --slots engine slots. Each is sent at its
measured time, closed loop per conversation: a turn waits for its predecessor's replayed prefill and keeps the
measured gap. Stats are over the profiling-phase requests that matched (send to prefill done, seconds). Every knob
takes a comma list; the replays are their product.
- --cache: slot (agentx_sim.SlotCache: tt-d-gen's whole-slot cache over the --slots slots), token (agentx_sim's
  token LRU over the same capacity, --slots x --slot-tokens), inf (unlimited), or engine (each request resumes where
  the engine resumed it, which isolates the timing model from the cache model).
- --policy: rr (tt-d-gen: chunks queued round robin over the seated requests, --inject-depth ahead of the device) or
  fcfs (a request's chunks in order).
- --request-overhead-ms / --request-overhead-ms-per-1k delay every request on its way to the engine, as in
  agentx_sim; --chunk-overhead-ms adds to every step.
- --chunk-time A,B: the run's box measured A ms + B us per prefix token for a full chunk; step time is scaled from
  --ref-chunk-time, the box the cost model was checked against, to it.
"""

import argparse
import bisect
import collections
import datetime as dt
import heapq
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_sim import BLOCK, PrefixCache, SlotCache, pct  # noqa: E402
from cost_model import Model  # noqa: E402

CHUNK = 8192
ANSI = re.compile(r"\x1b\[[0-9;]*m")
INJECT = re.compile(r"^(\S+) .*prefill inject slot_id=(\d+) start=(\d+) end=(\d+)")
ROUTE = re.compile(r'^(\S+)Z .*route request completed.* x_request_id="([^"]+)".* phase=(\w+)')
CACHES = ("slot", "token", "inf", "engine")
POLICIES = ("rr", "fcfs")


def utc(s):
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()


# ---- measured ----


def load_aiperf(path):
    """AIPerf's requests, in send order: id, conversation, phase, trace entry, prompt length, send and first token."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m, v = r["metadata"], {k: x["value"] for k, x in r["metrics"].items()}
            send = m["request_start_ns"] / 1e9
            out.append(
                dict(
                    rid=m["x_request_id"],
                    conv=m["conversation_id"],
                    phase=m["phase_kind"],
                    kind=m["source_kind"],
                    trace=m["source_trace_id"],
                    outer=m["source_outer_idx"],
                    isl=int(v["input_sequence_length"]),
                    send=send,
                    first=send + v["time_to_first_token"] / 1e3,
                )
            )
    return sorted(out, key=lambda r: r["send"])


def load_routes(path):
    """{x_request_id: {phase: time the frontend's route span for it completed}}."""
    routes = collections.defaultdict(dict)
    with open(path, errors="replace") as f:
        for line in f:
            if m := ROUTE.match(ANSI.sub("", line)):
                routes[m.group(2)][m.group(3)] = utc(m.group(1))
    return routes


def load_engine(path):
    """The prefill worker's requests, in first-inject order: (slot, first inject time, resume start, end, chunks). A
    request's chunks are consecutive injects on its slot; a chunk shorter than CHUNK is its last."""
    reqs, open_on = [], {}
    with open(path, errors="replace") as f:
        for line in f:
            m = INJECT.match(ANSI.sub("", line))
            if not m:
                continue
            t, slot, s, e = utc(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            cur = open_on.get(slot)
            if cur is None or cur[3] != s or cur[5]:
                cur = [slot, t, s, e, 0, False]
                reqs.append(cur)
                open_on[slot] = cur
            cur[3], cur[4], cur[5] = e, cur[4] + 1, e - s < CHUNK
    return [tuple(r[:5]) for r in reqs]


def match_engine(reqs, engine, routes):
    """Sets each request's prefill done (route_p: its Prefill route span) and engine request (eng: the first unused
    one of the same prompt length seated after its send). Returns the number of engine requests matched."""
    used = [False] * len(engine)
    times = [e[1] for e in engine]
    for r in reqs:
        r["route_p"] = routes.get(r["rid"], {}).get("Prefill")
        r["eng"] = None
        for j in range(bisect.bisect_left(times, r["send"]), len(engine)):
            if not used[j] and engine[j][3] == r["isl"]:
                used[j], r["eng"] = True, engine[j]
                break
    return sum(used)


def attach_traces(reqs, path):
    """Each request's session and block hashes, from its corpus entry (AIPerf's weka loader indexing)."""
    want = {r["trace"] for r in reqs}
    traces = {}
    with open(path) as f:
        for line in f:
            # The corpus is large: read each entry's id from the text and parse only the entries the run used.
            i = line.find('"id"')
            head = line[i : i + 80].split('"') if i >= 0 else []
            if len(head) > 3 and head[3] in want:
                t = json.loads(line)
                traces[t["id"]] = t["requests"]
    if missing := sorted(want - traces.keys()):
        raise SystemExit(f"{path}: no entry for {len(missing)} trace ids of the run, e.g. {', '.join(missing[:3])}")
    # AIPerf renders an :aux: sidecar so that it shares no prefix with any other conversation (the router saw zero
    # overlap for them on silicon), whatever its hash ids say: it gets a session of its own.
    keys = sorted(want | {r["conv"] for r in reqs if ":aux:" in r["conv"]})
    sess = {k: i for i, k in enumerate(keys)}
    for r in reqs:
        node = traces[r["trace"]][r["outer"]]
        if r["kind"] == "weka_subagent":
            # One subagent entry can become several child conversations and the export has no inner index: take the
            # inner request closest in length.
            node = min(node["requests"], key=lambda x: abs(r["isl"] - x["in"]))
        r["sess"] = sess[r["conv"] if ":aux:" in r["conv"] else r["trace"]]
        r["hashes"] = node["hash_ids"]


def load_run(run, traces):
    """The run's requests, matched to the engine and the corpus, and the number of engine requests matched."""
    reqs = load_aiperf(os.path.join(run, "aiperf", "profile_export.jsonl"))
    engine = load_engine(os.path.join(run, "prefill.log"))
    matched = match_engine(reqs, engine, load_routes(os.path.join(run, "frontend.log")))
    attach_traces(reqs, traces)
    return reqs, matched, len(engine)


def scored(reqs):
    """The requests the stats cover: profiling phase, with a prefill done and an engine request."""
    return [r for r in reqs if r["phase"] == "profiling" and r["route_p"] and r["eng"]]


# ---- caches: looked up when a request is seated, filled when its prefill is done ----


class Tokens:
    """agentx_sim.PrefixCache, unlimited (capacity None) or LRU over a token capacity."""

    def __init__(self, capacity=None):
        self.c = PrefixCache(capacity)

    def can_seat(self):
        return True

    def admit(self, r):
        return self.c.lookup(r["sess"], r["hashes"]), None

    def done(self, r, slot):
        self.c.insert(r["sess"], r["hashes"])


class Slots:
    """agentx_sim.SlotCache."""

    def __init__(self, n):
        self.c = SlotCache(n)

    def can_seat(self):
        return self.c.can_seat()

    def admit(self, r):
        return self.c.seat(r["sess"], r["hashes"], r["isl"])

    def done(self, r, slot):
        self.c.release(slot, r["hashes"])


class Engine(Tokens):
    """The engine's own resume points."""

    def admit(self, r):
        return (r["eng"][2] if r["eng"] else 0), None


def make_cache(name, slots, slot_tokens):
    return {
        "slot": lambda: Slots(slots),
        "token": lambda: Tokens(slots * slot_tokens),
        "inf": lambda: Tokens(),
        "engine": lambda: Engine(),
    }[name]()


# ---- replay ----


def scaled_steps(model, chunk_time, ref_chunk_time):
    """step_s with the step time scaled from the reference box's full-chunk time to the run's, both (ms, us/token)."""
    (a, b), (a0, b0) = chunk_time, ref_chunk_time

    def step_s(batch, C, budget):
        d, gap = model.step_s(batch, C, budget)
        pos = batch[0][1]
        return d * (a + b * 1e-3 * pos) / (a0 + b0 * 1e-3 * pos), gap

    return step_s


def replay(
    reqs,
    step_s,
    cache,
    policy,
    *,
    slots=6,
    align=1024,
    inject_depth=1,
    request_overhead_s=0.0,
    request_overhead_per_token_s=0.0,
    chunk_overhead_s=0.0,
):
    """({rid: (sent, resume start, prefill done)}, busy s) for reqs replayed through step_s(batch, C, budget).

    A conversation's first request is sent at its measured time; a later one when its predecessor's replayed prefill
    is done plus the measured gap from the predecessor's prefill done to its send (never before its measured send).
    A request reaches the engine its overhead after it is sent, waits for a slot, resumes at its cached prefix rounded
    down to align, and runs CHUNK-token chunks until its prompt is done."""
    prev, last = {}, {}
    for r in reqs:
        prev[r["rid"]], last[r["conv"]] = last.get(r["conv"]), r
    nxt = {p["rid"]: r for r in reqs if (p := prev[r["rid"]]) is not None}
    sent, out, heap = {}, {}, []

    def send(r, lag):
        sent[r["rid"]] = r["send"] + max(lag, 0.0)
        overhead = request_overhead_s + request_overhead_per_token_s * r["isl"]
        heapq.heappush(heap, (sent[r["rid"]] + overhead, r["send"], r["rid"], r))

    for r in reqs:
        if prev[r["rid"]] is None:
            send(r, 0.0)
    t = busy = 0.0
    pending, active, queued, turn = collections.deque(), [], collections.deque(), 0
    while heap or pending or active:
        while heap and heap[0][0] <= t:
            pending.append(heapq.heappop(heap)[3])
        while pending and len(active) < slots and cache.can_seat():
            r = pending.popleft()
            hit, slot = cache.admit(r)
            start = min(hit, r["isl"] - 1) // align * align
            active.append(dict(r=r, slot=slot, next=start, start=start))
        while len(queued) < inject_depth:
            ready = [a for a in active if a["next"] < a["r"]["isl"]]
            if not ready:
                break
            if policy == "rr":
                turn %= len(ready)
                a = ready[turn]
                turn += 1
            else:
                a = ready[0]
            queued.append((a, a["next"]))
            a["next"] += CHUNK
        if not queued:
            t = heap[0][0]
            continue
        a, pos = queued.popleft()
        d, gap = step_s([(CHUNK, pos)], CHUNK, CHUNK)
        d += chunk_overhead_s
        t += d
        busy += d
        r = a["r"]
        if pos + CHUNK >= r["isl"]:
            active.remove(a)
            cache.done(r, a["slot"])
            out[r["rid"]] = (sent[r["rid"]], a["start"], t)
            if r["rid"] in nxt:
                send(nxt[r["rid"]], t - (r["route_p"] or r["first"]))
        t += gap
    return out, busy


def stats(prefill_s, starts, isl):
    """Percentiles of prefill_s (send to prefill done) and the cache hit: resumed tokens over prompt tokens."""
    return dict(
        requests=len(prefill_s),
        p50=pct(prefill_s, 50),
        p90=pct(prefill_s, 90),
        p99=pct(prefill_s, 99),
        max=max(prefill_s),
        hit=sum(starts) / sum(isl),
    )


def measured(ok):
    return stats([r["route_p"] - r["send"] for r in ok], [r["eng"][2] for r in ok], [r["isl"] for r in ok])


def replayed(ok, out):
    rec = stats(
        [out[r["rid"]][2] - out[r["rid"]][0] for r in ok], [out[r["rid"]][1] for r in ok], [r["isl"] for r in ok]
    )
    rec["same_start"] = sum(out[r["rid"]][1] == r["eng"][2] for r in ok)
    return rec


# ---- CLI ----


def split(s, conv=str):
    return [conv(x) for x in str(s).split(",")]


def parse_chunk_time(s):
    a, b = split(s, float)
    if a <= 0 or b < 0:
        raise ValueError(s)
    return a, b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="directory with aiperf/profile_export.jsonl, frontend.log and prefill.log")
    ap.add_argument("--traces", required=True, help="the AgentX corpus the run was drawn from (traces.jsonl)")
    ap.add_argument("--cache", default="slot", help=f"one of {', '.join(CACHES)}")
    ap.add_argument("--policy", default="rr", help=f"one of {', '.join(POLICIES)}")
    ap.add_argument("--inject-depth", default="9", help="chunks queued ahead of the device (tt-d-gen on silicon: 9)")
    ap.add_argument("--request-overhead-ms", default="13.6", help="from send to reaching the engine (silicon: 13.6)")
    ap.add_argument("--request-overhead-ms-per-1k", default="0.87", help="per 1000 prompt tokens (silicon: 0.87)")
    ap.add_argument("--chunk-overhead-ms", default="0", help="added to every step")
    ap.add_argument("--slots", type=int, default=6, help="the engine's prefill slots")
    ap.add_argument("--slot-tokens", type=int, default=262144, help="tokens a slot holds")
    ap.add_argument("--align", type=int, default=1024, help="tokens a request's resume point is rounded down to")
    ap.add_argument("--chunk-time", help="A,B: the run's box ran a full chunk in A ms + B us per prefix token")
    ap.add_argument(
        "--ref-chunk-time",
        default="118.6,0.68",
        help="A,B for the box the cost model was checked against through tt-d-gen (tt-metal#60869)",
    )
    ap.add_argument("--json", help="write the measured and replayed stats here")
    a = ap.parse_args(argv)

    try:
        caches, policies = split(a.cache), split(a.policy)
        depths = split(a.inject_depth, int)
        overheads = [
            split(x, float) for x in (a.request_overhead_ms, a.request_overhead_ms_per_1k, a.chunk_overhead_ms)
        ]
    except ValueError as e:
        ap.error(str(e))
    for name, vals, ok in [("--cache", caches, CACHES), ("--policy", policies, POLICIES)]:
        if bad := [v for v in vals if v not in ok]:
            ap.error(f"bad {name} {','.join(bad)}: one of {', '.join(ok)}")
    if any(d < 1 for d in depths):
        ap.error(f"bad --inject-depth {a.inject_depth!r}: each must be at least 1")
    if any(v < 0 for vals in overheads for v in vals):
        ap.error("the overheads must not be negative")
    if a.slots < 1 or a.slot_tokens < 1 or a.align < 1:
        ap.error("--slots, --slot-tokens and --align must be positive")
    try:
        chunk_time = a.chunk_time and parse_chunk_time(a.chunk_time)
        ref_chunk_time = parse_chunk_time(a.ref_chunk_time)
    except ValueError:
        ap.error("--chunk-time and --ref-chunk-time take A,B with A > 0 and B >= 0")

    reqs, matched, n_engine = load_run(a.run, a.traces)
    ok = scored(reqs)
    if not ok:
        sys.exit(f"{a.run}: no profiling request has both a prefill route and an engine request")
    model = Model.load()
    step_s = scaled_steps(model, chunk_time, ref_chunk_time) if chunk_time else model.step_s
    print(f"{a.run}: {len(reqs)} requests, {n_engine} engine requests, {matched} matched; stats over {len(ok)}")
    print(f"{'':<46} {'p50':>7} {'p90':>7} {'p99':>7} {'max':>7} {'hit':>7}  same start")
    rows = [dict(name="measured", **measured(ok))]
    for cache, policy, depth, req_ms, per_1k, chunk_ms in itertools.product(caches, policies, depths, *overheads):
        c = make_cache(cache, a.slots, a.slot_tokens)
        out, busy = replay(
            reqs,
            step_s,
            c,
            policy,
            slots=a.slots,
            align=a.align,
            inject_depth=depth,
            request_overhead_s=req_ms / 1e3,
            request_overhead_per_token_s=per_1k / 1e6,
            chunk_overhead_s=chunk_ms / 1e3,
        )
        name = f"{cache} {policy} d{depth} {req_ms:g} ms + {per_1k:g} ms/1k" + (
            f" + {chunk_ms:g} ms/chunk" if chunk_ms else ""
        )
        rec = dict(name=name, cache=cache, policy=policy, inject_depth=depth, request_overhead_ms=req_ms)
        rec.update(request_overhead_ms_per_1k=per_1k, chunk_overhead_ms=chunk_ms, busy_s=busy, **replayed(ok, out))
        if isinstance(c, Slots):
            rec["evictions"] = c.c.evictions
        rows.append(rec)
    for r in rows:
        same = f"  {r['same_start']}/{len(ok)}" if "same_start" in r else ""
        print(f"{r['name']:<46} {r['p50']:7.3f} {r['p90']:7.3f} {r['p99']:7.3f} {r['max']:7.3f} {r['hit']:7.1%}{same}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(run=a.run, chunk_time=chunk_time or None, rows=rows), f, indent=1)
    return rows


if __name__ == "__main__":
    main()
