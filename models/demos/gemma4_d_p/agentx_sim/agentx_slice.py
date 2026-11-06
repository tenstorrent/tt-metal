#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""An AgentX slice for a silicon replay through the batching demo's scheduler, its cost-model prediction, and the
comparison with the measured run.

    agentx_slice.py build   --traces .../256k/traces.jsonl --out slice.json [--window 120 --start 600 --concurrency 48]
    agentx_slice.py predict --slice slice.json [--measured replay_results.json]

build: runs the closed-loop AgentX simulation (agentx_sim.run, 8k unbatched, infinite cache) and keeps the requests
sent inside [start, start + window): arrival (relative), input length and cached prefix. The replay then plays these
arrivals open loop, the same list for every mode.

predict: the batching demo scheduler's policy (demo/batching_scheduler.py, added with the continuous-batching PR), priced with cost_model:
- every request is padded to a multiple of 2048 (Scheduler.submit), and resumes at its cached prefix rounded down to
  its chunk width (the replay driver does the same);
- serial 8k / 4k / 2k: FCFS, one request at a time, one chunk per step, the last chunk padded;
- batched: FCFS admission into budget / 2048 = 4 KV slots, one per lane as in the demo, which the cost model is
  calibrated for; each active request gets one 2k lane per step, the spare lanes go to active requests in order (up to their remaining chunks - 1), and the active list rotates by one every step.
  This is the demo's policy, not agentx_sim's fcfs / pack, so it is priced here rather than through agentx_sim.run.
Step time = Model.step_s (calibrated on silicon per mode), plus the scheduler's gap between steps, which advances the
clock but, as in the replay's busy_s, is not busy time.
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_sim import load_sessions, pct, run  # noqa: E402
from cost_model import DEFAULT_DATA, Model  # noqa: E402

CHUNK = 2048
# mode -> (chunk width, token budget per step)
MODES = {"serial 8k": (8192, 8192), "serial 4k": (4096, 4096), "serial 2k": (2048, 2048), "batched": (2048, 8192)}


def build(a):
    model = Model.load(a.data)
    sessions = load_sessions(a.traces)
    res = run(
        model,
        sessions,
        8192,
        8192,
        "fcfs",
        a.concurrency,
        max_think_s=a.max_think_s,
        decode_tps=a.decode_tps,
        seed=a.seed,
        calibrated=False,  # the arrivals of the slices replayed so far
    )
    window = sorted((x.send, x.n_in, x.cached) for x in res["_stats"] if a.start <= x.send < a.start + a.window)
    if not window:
        last = max(x.send for x in res["_stats"])
        raise SystemExit(
            f"no requests sent in [{a.start:g}, {a.start + a.window:g}) s; the last was sent at {last:.0f} s"
        )
    reqs = [{"t": round(s - a.start, 4), "in": n_in, "cached": cached} for s, n_in, cached in window]
    with open(a.out, "w") as f:
        json.dump(
            {
                "source": os.path.abspath(a.traces),
                "concurrency": a.concurrency,
                "start": a.start,
                "window": a.window,
                "requests": reqs,
            },
            f,
            indent=1,
        )
    new = [r["in"] - r["cached"] for r in reqs]
    print(
        f"{len(reqs)} requests in {a.window:g} s; new tokens total {sum(new):,} (p50 {sorted(new)[len(new) // 2]:,}),"
        f" max input {max(r['in'] for r in reqs):,} -> {a.out}"
    )


def padded(n_in):
    return -(-n_in // CHUNK) * CHUNK


def start_of(r, width):
    return (min(r["cached"], r["in"] - 1) // width) * width


def predict(model, reqs, mode):
    """Per-request (arrival, start of service, finish) in seconds, plus busy time and step count."""
    width, budget = MODES[mode]
    batched = mode == "batched"
    num_slots = budget // width  # the demo's KV slots are its lanes, so the step budget fixes both
    limit = num_slots if batched else 1
    pending = collections.deque(sorted(range(len(reqs)), key=lambda i: reqs[i]["t"]))
    t, busy, steps = 0.0, 0.0, 0
    active = []  # [i, done, total] (done/total in tokens)
    started, finished = {}, {}
    while pending or active:
        while pending and reqs[pending[0]]["t"] <= t and len(active) < limit:
            i = pending.popleft()
            r = reqs[i]
            active.append([i, start_of(r, width), padded(r["in"])])
            started[i] = t
        if not active:
            t = reqs[pending[0]]["t"]
            continue
        if batched:
            plan = [[e, 1] for e in active]
            free = num_slots - len(plan)
            for p in plan:
                left = -(-(p[0][2] - p[0][1]) // width)
                extra = min(free, left - 1)
                p[1] += extra
                free -= extra
            active.append(active.pop(0))
        else:
            plan = [[active[0], 1]]
        lanes = [(width, e[1] + j * width) for e, n in plan for j in range(n)]
        dt, gap = model.step_s(lanes, width, budget)
        t += dt
        busy += dt
        steps += 1
        for e, n in plan:
            e[1] += n * width
            if e[1] >= e[2]:
                finished[e[0]] = t
                active.remove(e)
        t += gap
    return started, finished, busy, steps


def summary(reqs, started, finished):
    ttft = [finished[i] - reqs[i]["t"] for i in finished]
    small = [finished[i] - reqs[i]["t"] for i in finished if reqs[i]["in"] - reqs[i]["cached"] <= 4096]
    return {
        "n": len(ttft),
        "p50": pct(ttft, 50),
        "p90": pct(ttft, 90),
        "p99": pct(ttft, 99),
        "mean": sum(ttft) / len(ttft),
        "small_p50": pct(small, 50),
        "makespan": max(finished.values()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "predict"])
    ap.add_argument("--traces")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default="agentx_slice.json")
    ap.add_argument("--slice")
    ap.add_argument("--measured", help="the replay's results JSON, to print next to the prediction")
    ap.add_argument("--window", type=float, default=120.0)
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-think-s", type=float, default=60.0)
    ap.add_argument("--decode-tps", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a)
    model = Model.load(a.data)
    with open(a.slice) as f:
        reqs = json.load(f)["requests"]
    meas = {}
    if a.measured:
        with open(a.measured) as f:
            meas = json.load(f)["modes"]
    print(f"{len(reqs)} requests; batched slots {MODES['batched'][1] // CHUNK}")
    print(
        f"{'mode':>10} | {'pred p50':>8} {'p90':>6} {'p99':>6} {'small50':>7} {'makespan':>8} {'busy':>6} {'steps':>5}"
        + (
            f" | {'meas p50':>8} {'p90':>6} {'p99':>6} {'small50':>7} {'makespan':>8} {'busy':>6} {'steps':>5} {'step err':>8}"
            if meas
            else ""
        )
    )
    for mode in MODES:
        st, fi, busy, steps = predict(model, reqs, mode)
        p = summary(reqs, st, fi)
        line = (
            f"{mode:>10} | {p['p50']:8.3f} {p['p90']:6.2f} {p['p99']:6.2f} {p['small_p50']:7.3f} {p['makespan']:8.2f} "
            f"{busy:6.2f} {steps:5d}"
        )
        if mode in meas:
            m = meas[mode]
            mr = {int(k): v for k, v in m["requests"].items()}
            mst = {i: v["started"] for i, v in mr.items()}
            mfi = {i: v["finished"] for i, v in mr.items()}
            q = summary(reqs, mst, mfi)
            err = (m["busy_s"] / m["steps"]) / (busy / steps) - 1 if m["steps"] else float("nan")
            line += (
                f" | {q['p50']:8.3f} {q['p90']:6.2f} {q['p99']:6.2f} {q['small_p50']:7.3f} {q['makespan']:8.2f} "
                f"{m['busy_s']:6.2f} {m['steps']:5d} {100 * err:+7.1f}%"
            )
        print(line)


if __name__ == "__main__":
    main()
