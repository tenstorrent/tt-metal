#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AgentX replay of Gemma4 prefill mesh designs: one SP8 Galaxy vs two SP4 halves (host only).

    agentx_designs.py --traces .../traces.jsonl --designs sp8_lanes:1x4,sp4_lanes:2x2,sp4_lanes:2x4 \
        --concurrency 32,64,96,128 --cache inf,2.2e6 --policy fcfs,pack [--json runs.jsonl]

A design spec is NAME:SERVERSxSLOTS: NAME a design in sp_designs_1028.json (designs.py), SERVERS prefill servers per
Galaxy, SLOTS 2k lanes per step per server. Each server is agentx_sim.run() unchanged, priced with that design's
measured step table: server k serves every SERVERS-th session (a session's KV stays where it was prefilled), gets
1 / SERVERS of the concurrency and of the prefix cache. TTFT percentiles are over every request of every server;
steady-state useful tok/s is the servers' sum.
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentx_sim  # noqa: E402
from designs import DEFAULT_DESIGNS, DesignModel  # noqa: E402

C = 2048
G = {}


def parse_spec(spec):
    name, shape = spec.split(":")
    servers, slots = (int(x) for x in shape.split("x"))
    return name, servers, slots


def run_spec(job):
    spec, conc, cache, policy, think = job
    name, servers, slots = parse_spec(spec)
    model = DesignModel.load(name, G["designs"])
    stats, ss = [], []
    for k in range(servers):
        r = agentx_sim.run(
            model,
            G["sessions"][k::servers],
            C,
            C * slots,
            policy,
            max(1, conc // servers),
            cache_tokens=None if cache == "inf" else int(float(cache) / servers),
            max_think_s=think,
            decode_tps=G["decode_tps"],
        )
        stats += r["_stats"]
        ss.append(r)
    pct = agentx_sim.pct
    ttft = [x.done - x.send for x in stats]
    bucket = lambda lo, hi: [x.done - x.send for x in stats if lo < x.n_in - x.cached <= hi]
    small, big = bucket(-1, 4096), bucket(16384, 1 << 40)
    return dict(
        design=spec,
        concurrency=conc,
        cache=cache,
        policy=policy,
        think=think,
        requests=len(stats),
        ttft_p50=pct(ttft, 50),
        ttft_p90=pct(ttft, 90),
        ttft_p99=pct(ttft, 99),
        small_p50=pct(small, 50),
        small_p90=pct(small, 90),
        big_p50=pct(big, 50),
        big_p90=pct(big, 90),
        hit=sum(x.cached for x in stats) / sum(x.n_in for x in stats),
        ss_util=sum(r.get("ss_util", float("nan")) for r in ss) / servers,
        ss_useful_tok_s=sum(r.get("ss_useful_ideal_tok_s", float("nan")) for r in ss),
        ss_ttft_p50=sum(r.get("ss_ttft_p50", float("nan")) for r in ss) / servers,
        makespan_s=max(r["makespan_s"] for r in ss),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--designs-file", default=DEFAULT_DESIGNS)
    ap.add_argument("--designs", default="sp8_lanes:1x4,sp8_loop:1x4,sp4_lanes:2x2,sp4_loop:2x2")
    ap.add_argument("--concurrency", default="32,64,96,128")
    ap.add_argument("--cache", default="inf,2.2e6")
    ap.add_argument("--policy", default="fcfs,pack")
    ap.add_argument("--max-think-s", default="60")
    ap.add_argument("--decode-tps", type=float, default=100.0)
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--json", help="append one JSON line per run here")
    a = ap.parse_args()
    split = lambda s, f=str: [f(x) for x in s.split(",") if x]
    G.update(designs=a.designs_file, decode_tps=a.decode_tps)
    for spec in split(a.designs):
        name, _, slots = parse_spec(spec)
        model = DesignModel.load(name, a.designs_file)
        if slots > model.max_lanes:
            raise SystemExit(f"{spec}: {name} is measured up to {model.max_lanes} lanes")
    G["sessions"] = agentx_sim.load_sessions(a.traces)
    jobs = list(
        itertools.product(
            split(a.designs), split(a.concurrency, int), split(a.cache), split(a.policy), split(a.max_think_s, float)
        )
    )
    hdr = (
        f"{'design':>16} {'conc':>4} {'cache':>6} {'pol':>4} {'think':>5} | {'p50':>6} {'p90':>6} {'p99':>7} | "
        f"{'<=4K p90':>8} {'>16K p50':>8} {'>16K p90':>8} | {'ss util':>7} {'ss useful':>9} {'hit':>5}"
    )
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    with mp.get_context("fork").Pool(a.j) as pool:
        for r in pool.imap(run_spec, jobs):
            print(
                f"{r['design']:>16} {r['concurrency']:>4} {r['cache']:>6} {r['policy']:>4} {r['think']:5g} | "
                f"{r['ttft_p50']:6.2f} {r['ttft_p90']:6.2f} {r['ttft_p99']:7.1f} | {r['small_p90']:8.2f} "
                f"{r['big_p50']:8.2f} {r['big_p90']:8.1f} | {r['ss_util']:7.2f} {r['ss_useful_tok_s'] / 1e3:8.1f}k "
                f"{r['hit']:5.3f}",
                flush=True,
            )
            if a.json:
                with open(a.json, "a") as f:
                    f.write(json.dumps(agentx_sim.json_record(r), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
