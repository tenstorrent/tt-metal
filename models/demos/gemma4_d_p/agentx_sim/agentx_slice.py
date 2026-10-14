#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""An AgentX slice for a silicon replay through the batching demo's scheduler, its cost-model prediction, and the
comparison with the measured run.

    agentx_slice.py build   --traces .../256k/traces.jsonl --out slice.json [--window 120 --start 600 --concurrency 48]
    agentx_slice.py predict --slice slice.json
    agentx_slice.py compare --slice slice.json --measured replay_results.json

build: runs the closed-loop AgentX simulation (agentx_sim.run, 8k unbatched, infinite cache) and keeps the requests
sent inside [start, start + window): arrival (relative), input length and cached prefix. The replay then plays these
arrivals open loop, the same list for every mode.

predict: the demo scheduler's policy (models/demos/gemma4_d_p/demo/batching_server.py), priced with cost_model:
- every request is padded to a multiple of 2048 (Scheduler.submit), and resumes at its cached prefix rounded down to
  its chunk width (the replay driver does the same);
- serial 8k / 4k / 2k: FCFS, one request at a time, one chunk per step, the last chunk padded;
- batched: FCFS admission into num_slots KV slots; each active request gets one 2k lane per step, the spare lanes go
  to active requests in order (up to their remaining chunks - 1), and the active list rotates by one every step.
Step time = Model.step (pessimistic attention) + a host overhead per step (--host-ms, default 0: device time only).
--calibrated applies the silicon-measured device scale and host time per mode (CAL_SCALE, CAL_HOST_MS).
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_sim import load_sessions, run  # noqa: E402
from cost_model import Model  # noqa: E402

CHUNK = 2048
MODES = {"serial 8k": 8192, "serial 4k": 4096, "serial 2k": 2048, "batched": 2048}
# Measured on silicon (bh-glx-110-d06u02, main 836bf0b + the batching PoC, three AgentX slices): device time per
# step over the cost model, and host time per step (busy minus trace replay).
CAL_SCALE = {"serial 8k": 1.125, "serial 4k": 1.057, "serial 2k": 1.009, "batched": 1.054}
CAL_HOST_MS = {"serial 8k": 3.3, "serial 4k": 3.3, "serial 2k": 3.3, "batched": 6.3}


def build(a):
    model = Model(json.load(open(a.data)))
    sessions = load_sessions(a.traces)
    res = run(
        model, sessions, 8192, 8192, "fcfs", a.concurrency, None, "chunk", a.max_think_s, a.decode_tps, seed=a.seed
    )
    # stats rows: (send, done, in, cached, start, rows)
    window = sorted((s, n_in, cached) for s, _, n_in, cached, *_ in res["_stats"] if a.start <= s < a.start + a.window)
    reqs = [{"t": round(s - a.start, 4), "in": n_in, "cached": cached} for s, n_in, cached in window]
    json.dump(
        {
            "source": os.path.abspath(a.traces),
            "concurrency": a.concurrency,
            "start": a.start,
            "window": a.window,
            "requests": reqs,
        },
        open(a.out, "w"),
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


def predict(model, reqs, mode, num_slots=4, host_ms=0.0, scale=None):
    """Per-request (arrival, start of service, finish) in seconds, plus busy time and step count."""
    width = MODES[mode]
    batched = mode == "batched"
    pending = collections.deque(sorted(range(len(reqs)), key=lambda i: reqs[i]["t"]))
    t, busy, steps = 0.0, 0.0, 0
    active = []  # [i, done, total] (done/total in tokens)
    started, finished = {}, {}
    while pending or active:
        limit = num_slots if batched else 1
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
        dt = model.step(lanes, "pess")["total"] * (scale or {}).get(mode, 1.0) / 1e6 + host_ms / 1e3
        t += dt
        busy += dt
        steps += 1
        for e, n in plan:
            e[1] += n * width
            if e[1] >= e[2]:
                finished[e[0]] = t
                active.remove(e)
    return started, finished, busy, steps


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


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
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "predict", "compare"])
    ap.add_argument("--traces")
    ap.add_argument("--data", default=os.path.join(here, "model_data_revc.json"))
    ap.add_argument("--out", default="agentx_slice.json")
    ap.add_argument("--slice")
    ap.add_argument("--measured")
    ap.add_argument("--window", type=float, default=120.0)
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-think-s", type=float, default=60.0)
    ap.add_argument("--decode-tps", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--host-ms", type=float, default=0.0)
    ap.add_argument(
        "--calibrated",
        action="store_true",
        help="device scale and host ms per step per mode from the silicon replay (CAL_SCALE, CAL_HOST_MS); overrides --host-ms",
    )
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a)
    model = Model(json.load(open(a.data)))
    reqs = json.load(open(a.slice))["requests"]
    meas = json.load(open(a.measured))["modes"] if a.cmd == "compare" else {}
    host = "per mode (calibrated)" if a.calibrated else f"{a.host_ms:g} ms/step"
    print(f"{len(reqs)} requests; slots {a.slots}; host overhead {host} in the prediction")
    print(
        f"{'mode':>10} | {'pred p50':>8} {'p90':>6} {'p99':>6} {'small50':>7} {'makespan':>8} {'busy':>6} {'steps':>5}"
        + (
            f" | {'meas p50':>8} {'p90':>6} {'p99':>6} {'small50':>7} {'makespan':>8} {'busy':>6} {'steps':>5} {'step err':>8}"
            if meas
            else ""
        )
    )
    for mode in MODES:
        scale = CAL_SCALE if a.calibrated else None
        host_ms = CAL_HOST_MS[mode] if a.calibrated else a.host_ms
        st, fi, busy, steps = predict(model, reqs, mode, a.slots, host_ms, scale)
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
