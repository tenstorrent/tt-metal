#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Continuous-batching cost model for Gemma4 chunked prefill (BH Galaxy 8x4, CP8 / TP4).

    step(batch) = dense(M_total) + sum_r attn(C_r, P_r) + fixed

per layer type, times 10 global + 50 sliding layers. M = rows per device = sum(C_r) / CP.
Data lives in a JSON file (default: model_data_revc.json next to this script) so new sweeps only
update numbers. Usage:

    cost_model.py [--data model_data_revc.json] fits      # coefficients + reconstruction check
    cost_model.py [--data ...] policy                # chunk unit x token budget tables
    cost_model.py [--data ...] mixed                 # one long prompt among short ones
"""

import argparse
import bisect
import json
import math
import os

CP = 8
N_LAYERS = {"global": 10, "sliding": 50}
WINDOW = 1024
# Per device, per layer. Global: 8 local Q heads x head_dim 512; sliding: 8 x 256.
GLOBAL_FLOP_PER_QK = 4 * 8 * 512  # x (C/CP) x (P + C/2)
SLIDING_FLOP_PER_QK = 4 * 8 * 256  # x (C/CP) x WINDOW


def global_flop(C, P):
    return GLOBAL_FLOP_PER_QK * (C / CP) * (P + C / 2)


def sliding_flop(C):
    return SLIDING_FLOP_PER_QK * (C / CP) * WINDOW


class Model:
    def __init__(self, data):
        self.d = data
        # dense(M) per layer type: piecewise linear through measured M points (no extrapolation past the ends).
        self.dense_pts = {L: sorted((int(m), v) for m, v in data["dense_us"][L].items()) for L in N_LAYERS}
        # attn_global(C, P) = a_C + b_C * P per layer, from measured (P, us) points by least squares.
        self.g = {}
        for C, pts in data["gattn_us"].items():
            xs = [p for p, _ in pts]
            ys = [t for _, t in pts]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
            self.g[int(C)] = (my - b * mx, b, pts)
        self.l = {int(C): v for C, v in data["lattn_us"].items()}
        self.fixed_us = data.get("fixed_step_us", 0.0)
        o = data["optimistic"]
        self.opt_g_rate, self.opt_g_fixed = o["global_tflops"] * 1e12, o["global_fixed_us"]
        self.opt_l_us_per_gflop, self.opt_l_fixed = o["sliding_us_per_gflop"], o["sliding_fixed_us"]

    # ---- components (per layer, us) ----
    def dense(self, L, M):
        pts = self.dense_pts[L]
        ms = [m for m, _ in pts]
        if M <= ms[0]:
            # Below the smallest measured M: hold flat (rows are free when weights dominate); flagged in reports.
            return pts[0][1]
        if M >= ms[-1]:
            if M > ms[-1]:
                raise ValueError(f"M={M} past the last measured point {ms[-1]}; not extrapolating")
            return pts[-1][1]
        i = bisect.bisect_left(ms, M)
        (m0, v0), (m1, v1) = pts[i - 1], pts[i]
        return v0 + (v1 - v0) * (M - m0) / (m1 - m0)

    def gattn(self, C, P):
        if C not in self.g:  # a multi-slot piece with no measured chunk size: run it as unit-size chunks
            u = max(c for c in self.g if C % c == 0)
            return sum(self.gattn(u, P + j * u) for j in range(C // u))
        a, b, _ = self.g[C]
        return a + b * P

    def lattn(self, C):
        if C not in self.l:
            u = max(c for c in self.l if C % c == 0)
            return (C // u) * self.l[u]
        return self.l[C]

    # ---- one step ----
    def step(self, batch, mode="pess"):
        """batch: list of (C, P). Returns dict of per-step us by component (all layers summed)."""
        M = sum(C for C, _ in batch) // CP
        out = {"dense": sum(N_LAYERS[L] * self.dense(L, M) for L in N_LAYERS), "fixed": self.fixed_us}
        if mode == "pess":
            out["gattn"] = N_LAYERS["global"] * sum(self.gattn(C, P) for C, P in batch)
            out["lattn"] = N_LAYERS["sliding"] * sum(self.lattn(C) for C, _ in batch)
        else:  # one batched call per layer at the big-chunk rate
            g = sum(global_flop(C, P) for C, P in batch)
            out["gattn"] = N_LAYERS["global"] * (self.opt_g_fixed + g / self.opt_g_rate * 1e6)
            lf = sum(sliding_flop(C) for C, _ in batch) / 1e9
            out["lattn"] = N_LAYERS["sliding"] * (self.opt_l_fixed + lf * self.opt_l_us_per_gflop)
            # The optimistic bound can't be slower than running the same requests standalone.
            pess = self.step(batch, "pess")
            out["gattn"], out["lattn"] = min(out["gattn"], pess["gattn"]), min(out["lattn"], pess["lattn"])
        out["total"] = sum(out.values())
        return out

    def unbatched_prompt_s(self, C, tokens):
        n = math.ceil(tokens / C)
        return sum(self.step([(C, i * C)])["total"] for i in range(n)) / 1e6


# ---- scheduler ----
def simulate(model, reqs, C, budget, mode="pess", policy="fcfs", spill=False):
    """reqs: list of (arrival_s, prompt_tokens). Every step packs up to budget/C slots, one chunk per request.
    policy: fcfs (admit in arrival order, keep slots until done) | srpt (shortest remaining first, per step) |
    pack (shortest remaining first, each request takes every slot it still needs: long prompts run as big chunks).
    spill: slots left empty go to running requests (most remaining first), so a lone request runs as a bigger chunk.
    Returns per-request TTFT (s, from arrival), makespan, and component totals."""
    slots = budget // C
    t = 0.0
    pending = sorted(range(len(reqs)), key=lambda i: reqs[i][0])
    done_at = [None] * len(reqs)
    progress = [0] * len(reqs)  # tokens prefilled
    need = [math.ceil(r[1] / C) * C for r in reqs]  # last chunk padded to C (fixed trace shapes)
    comp = {}
    active = []
    steps = 0
    rows_used = 0
    while pending or active:
        arrived = [i for i in pending if reqs[i][0] <= t]
        if policy == "fcfs":
            while len(active) < slots and arrived:
                i = arrived.pop(0)
                pending.remove(i)
                active.append(i)
            run = active
        elif policy == "pack":  # shortest remaining first; each request takes every slot it still needs
            for i in arrived:
                pending.remove(i)
            active = sorted(active + arrived, key=lambda i: need[i] - progress[i])
            run, free = [], slots
            for i in active:
                if free == 0:
                    break
                run.append(i)
                free -= min(free, (need[i] - progress[i]) // C)
        else:  # srpt over everything arrived and active
            for i in arrived:
                pending.remove(i)
            active = sorted(active + arrived, key=lambda i: need[i] - progress[i])
            run = active[:slots]
        if not run:
            t = reqs[pending[0]][0]
            continue
        take = {i: 1 for i in run}
        if policy == "pack":
            free = slots
            for i in run:
                take[i] = min(free, (need[i] - progress[i]) // C)
                free -= take[i]
        elif spill:
            free = slots - len(run)
            while free > 0:
                left = [i for i in run if (need[i] - progress[i]) // C > take[i]]
                if not left:
                    break
                i = max(left, key=lambda j: need[j] - progress[j] - take[j] * C)
                take[i] += 1
                free -= 1
        batch = [(take[i] * C, progress[i]) for i in run]
        s = model.step(batch, mode)
        for k, v in s.items():
            comp[k] = comp.get(k, 0.0) + v
        t += s["total"] / 1e6
        steps += 1
        rows_used += sum(take.values())
        for i in run:
            progress[i] += take[i] * C
            if progress[i] >= need[i]:
                done_at[i] = t
        active = [i for i in active if done_at[i] is None]
    ttft = [done_at[i] - reqs[i][0] for i in range(len(reqs))]
    return {"ttft": ttft, "makespan": t, "comp": comp, "steps": steps, "slot_fill": rows_used / (steps * slots)}


# ---- reports ----
def report_fits(m):
    print("dense(M) per layer, us (M = rows per device):")
    for L in N_LAYERS:
        print(f"  {L:8s} " + "  ".join(f"M={mm}: {v:.0f}" for mm, v in m.dense_pts[L]))
    pts = {L: m.dense_pts[L] for L in N_LAYERS}
    ms = [mm for mm, _ in pts["sliding"]]
    print("dense per step, ms:  " + "  ".join(f"M={mm}: {m.step([(mm * CP, 0)])['dense'] / 1e3:.1f}" for mm in ms))
    for i in range(1, len(ms)):
        m0, m1 = ms[i - 1], ms[i]
        d0, d1 = (m.step([(x * CP, 0)])["dense"] for x in (m0, m1))
        print(
            f"   marginal {m0}->{m1}: {(d1 - d0) / (m1 - m0):.1f} us per row ({(d1 - d0) / (m1 - m0) / 8 * 1e3:.0f} ns per token)"
        )
    print("attn_global(C, P) = a + b*P per layer:")
    for C, (a, b, pts_) in sorted(m.g.items()):
        rate = GLOBAL_FLOP_PER_QK * (C / CP) / (b * 1e-6) / 1e12 if b else float("nan")
        resid = max(abs(a + b * p - t) for p, t in pts_)
        print(
            f"  C={C:5d}: a={a:7.0f} us  b={b * 1e3:6.2f} us per 1k prefix tokens  slope rate {rate:5.0f} TFLOP/s"
            f"  ({len(pts_)} pts, max resid {resid:.0f} us)"
        )
    print("attn_sliding(C) per layer, us: " + "  ".join(f"C={C}: {v:.0f}" for C, v in sorted(m.l.items())))
    print("unbatched reconstruction (device time, s) vs measured:")
    for C, meas in m.d.get("measured_unbatched", {}).items():
        C = int(C)
        if C not in m.g or C not in m.l:
            continue
        first = m.step([(C, 0)])["total"] / 1e3
        r100 = m.unbatched_prompt_s(C, 13 * 8192)
        r256 = m.unbatched_prompt_s(C, 262144)
        print(
            f"  C={C}: first {first:.1f} ms (meas {meas.get('first_ms')})  106k {r100:.2f} s (meas {meas.get('mid_s')})"
            f"  256k {r256:.2f} s (meas {meas.get('full_s')})"
        )


def report_policy(m, units=(1024, 2048, 4096), budgets=(2048, 4096, 8192, 16384), prompt=32768, nreq=16):
    print(f"\nOffline batch: {nreq} prompts of {prompt} tokens, all queued at t=0, FCFS.")
    print(
        "unit  budget  slots | makespan s  tok/s     x vs unbatched-same-unit  | mean TTFT s  | step ms first  | mode"
    )
    for C in units:
        if C not in m.l:
            continue
        base = m.unbatched_prompt_s(C, prompt) * nreq
        for B in budgets:
            if B < C or (B // CP) > max(mm for mm, _ in m.dense_pts["sliding"]):
                continue
            for mode in ("pess", "opt"):
                r = simulate(m, [(0.0, prompt)] * nreq, C, B, mode)
                tp = nreq * prompt / r["makespan"]
                first = m.step([(C, 0)] * (B // C), mode)["total"] / 1e3
                print(
                    f"{C:5d} {B:6d} {B // C:5d} | {r['makespan']:9.2f} {tp:9.0f} {base / r['makespan']:6.2f}x"
                    f"                 | {sum(r['ttft']) / nreq:8.2f}     | {first:8.1f}       | {mode}"
                )


def report_mixed(m, C=2048, B=8192):
    long = (0.0, 262144)
    shorts = [(0.05 * i, 4096) for i in range(40)]
    reqs = [long] + shorts
    print(f"\nMixed: one 256k prompt at t=0 plus 40 x 4k prompts arriving every 50 ms; unit {C}, budget {B}.")
    for pol, spill in (("fcfs", False), ("srpt", False), ("fcfs", True), ("srpt", True)):
        r = simulate(m, reqs, C, B, "pess", pol, spill)
        sh = sorted(r["ttft"][1:])
        print(
            f"  {pol}{'+spill' if spill else '      '}: long TTFT {r['ttft'][0]:.2f} s | short TTFT p50 {sh[len(sh) // 2]:.2f} p95 {sh[int(len(sh) * .95)]:.2f} s"
            f" | makespan {r['makespan']:.2f} s | slot fill {r['slot_fill']:.0%}"
        )
    solo = m.unbatched_prompt_s(C, 262144)
    print(f"  (256k alone, unbatched at {C}: {solo:.2f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data_revc.json"))
    ap.add_argument("what", nargs="*", default=["fits", "policy", "mixed"])
    a = ap.parse_args()
    m = Model(json.load(open(a.data)))
    for w in a.what:
        {"fits": report_fits, "policy": report_policy, "mixed": report_mixed}[w](m)


if __name__ == "__main__":
    main()
