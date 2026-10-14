#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep agentx_sim over a grid of knobs in parallel, one JSON line per run (with its wall time).

    agentx_sweep.py --traces .../256k/traces.jsonl --out sweep.jsonl -j 12 \
        --config 8192x8192:fcfs,2048x8192:pack --concurrency 16,64,256 --cache inf,4e6 [--align chunk,exact] \
        [--max-think-s 60] [--decode-tps 100] [--pieces lanes] [--mode pess]

Every list knob is comma-separated; the grid is their product. The traces are parsed once in the parent and shared
with the forked workers. Prints one summary row per finished run and the total wall time.
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_sim import load_sessions, run  # noqa: E402
from cost_model import Model  # noqa: E402

G = {}


def one(job):
    cfg, conc, cache, align, think, dtps, pieces, mode, host_ms, scale = job
    cb, policy = cfg.split(":")
    C, B = (int(x) for x in cb.split("x"))
    cache_tokens = None if cache == "inf" else int(float(cache))
    t = time.time()
    r = run(
        G["model"],
        G["sessions"],
        C,
        B,
        policy,
        conc,
        cache_tokens,
        align,
        think,
        dtps,
        0.0,
        mode,
        pieces=pieces,
        host_ms=host_ms,
        scale=scale,
    )
    r = {k: v for k, v in r.items() if not k.startswith("_")}
    return {
        "config": cfg,
        "concurrency": conc,
        "cache": cache,
        "align": align,
        "max_think_s": think,
        "decode_tps": dtps,
        "pieces": pieces,
        "mode": mode,
        "host_ms": host_ms,
        "scale": scale,
        "wall_s": round(time.time() - t, 2),
        **r,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--data", default=os.path.join(here, "model_data_revc.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--config", default="8192x8192:fcfs,2048x8192:pack")
    ap.add_argument("--concurrency", default="64")
    ap.add_argument("--cache", default="inf")
    ap.add_argument("--align", default="chunk")
    ap.add_argument("--max-think-s", default="60")
    ap.add_argument("--decode-tps", default="100")
    ap.add_argument("--pieces", default="lanes")
    ap.add_argument("--mode", default="pess")
    ap.add_argument("--host-ms", default="0")
    ap.add_argument("--scale", default="1")
    a = ap.parse_args()
    t0 = time.time()
    G["model"] = Model(json.load(open(a.data)))
    G["sessions"] = load_sessions(a.traces)
    t_load = time.time() - t0
    split = lambda s: [x for x in s.split(",") if x]
    jobs = list(
        itertools.product(
            split(a.config),
            [int(x) for x in split(a.concurrency)],
            split(a.cache),
            split(a.align),
            [float(x) for x in split(a.max_think_s)],
            [float(x) for x in split(a.decode_tps)],
            split(a.pieces),
            split(a.mode),
            [float(x) for x in split(a.host_ms)],
            [float(x) for x in split(a.scale)],
        )
    )
    print(f"{len(jobs)} runs on {a.j} workers; traces loaded in {t_load:.1f} s", flush=True)
    with mp.get_context("fork").Pool(a.j) as pool, open(a.out, "a") as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            ss = (
                f"ss useful {r['ss_useful_ideal_tok_s'] / 1e3:5.1f}k util {r['ss_util']:.2f} p50 {r['ss_ttft_p50']:6.2f}"
                if "ss_util" in r
                else "no steady window"
            )
            print(
                f"{r['config']:>16} c{r['concurrency']:<4} cache {r['cache']:>5} {r['align']:>5} think {r['max_think_s']:g}"
                f" | hit {r['hit_rate']:.3f} used {r['hit_rate_used']:.3f} evict-recompute {r['evict_recompute_pct']:4.1f}% | {ss} | {r['wall_s']:6.1f} s",
                flush=True,
            )
    print(f"total {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
