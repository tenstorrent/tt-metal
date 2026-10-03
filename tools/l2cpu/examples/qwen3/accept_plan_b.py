#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Acceptance of the device-resident loop (Plan B) on the sampling firmware against the host C reference
(decode_harness hostref sampler, host-mediated loop, same process): token lists must be identical.

    tools/l2cpu/scripts/l2cpu_run.sh "accept" $PY tools/l2cpu/examples/qwen3/accept_plan_b.py --model Qwen/Qwen3-8B --batch 1 --prompts 5 --steps 256
batch 32: per-user mixed settings from sampling_mix.py (user u: setting u % 4, seed 1234 + u), one group of 32 prompts.
--tiles 2 / 4: the batch split over L2CPU tiles (contiguous user blocks); the reference is the same host C library
run (global user index), so a split run must be identical to it.
Same model, prompts, prefill, per-user params, seeds and step convention in both paths (plan_b.run_group_planb).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)

SETTINGS = {
    "greedy": (0.0, 0, 1.0),
    "T0.7k50p0.9": (0.7, 50, 0.9),
    "T1.0k0p1.0": (1.0, 0, 1.0),
    "T0.6k20p0.95": (0.6, 20, 0.95),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--negative-control",
        action="store_true",
        help="perturb ONLY the x280 side: seed + 1 of user 1 (batch 32) / user 0 (batch 1); pass iff exactly that user diverges",
    )
    ap.add_argument(
        "--neg-user", type=int, default=None, help="--negative-control: the user to perturb (default 1 / 0)"
    )
    ap.add_argument(
        "--repeat", type=int, default=1, help="Plan B runs per group (soak), each compared to the one reference"
    )
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--tiles", type=int, default=1, choices=[1, 2, 4], help="L2CPU tiles serving the batch (batch > 1)")
    ap.add_argument("--inject-tile", type=int, default=0, help="tile index (into the started tiles) for the test hooks")
    ap.add_argument(
        "--inject-trap", type=int, default=None, help="hart: firmware trap test during the first Plan B run"
    )
    ap.add_argument("--inject-at", type=int, default=64, help="decode step at which the trap is injected")
    ap.add_argument(
        "--retry-on-timeout",
        type=int,
        default=0,
        metavar="CHUNK",
        help="serving-style recovery: enqueue CHUNK steps at a time; on a wait timeout / firmware error, "
        "warm-restart the firmware and retry the failed step (needs a restartable image)",
    )
    ap.add_argument(
        "--inject-timeout-at",
        type=int,
        default=None,
        help="test hook for --retry-on-timeout: trap hart 0 through the mailbox at this decode step of "
        "every group (the step's wait times out); tokens must still match the reference",
    )
    ap.add_argument(
        "--restart-every",
        type=int,
        default=0,
        help="WARM restart of the firmware every N decode steps while the traces run (alternating L1 "
        "mailbox PARK and L2 RNMI park); needs X280_RESTARTABLE=1 and a restartable image",
    )
    a = ap.parse_args()
    import decode_harness as dh

    home = dh.check_env()
    os.chdir(home)
    os.environ["HF_MODEL"] = a.model
    os.environ["TT_CACHE_PATH"] = deps.weights_cache(a.model)
    import torch

    torch.manual_seed(213919)
    from dh_samplers import make_sampler
    from plan_b import PlanB, run_group_planb
    from l2cpu_monitor import SamplingMonitor
    from l2cpu_sampler import session

    class NS(types.SimpleNamespace):
        def __getattr__(self, k):
            return None

    B, N = a.batch, a.steps
    ns = NS(
        trace_region_size=200_000_000,
        precision="performance",
        batch=B,
        max_seq_len=1024,
        prefill_mode="batched",
        convert="oneshot",
        one_trace=False,
        input_tokens=None,
        clear_kv=1,
        max_new_tokens=N,
        hash_rows=0,
        x280=True,
        x280_tiles=a.tiles,
    )
    h = dh.DecodeHarness(ns)  # x280=True: arena + firmware boot before any capture
    pb = PlanB(h.mesh, session=session(), uncached=B > 1, log=dh.log)
    fws = session().get("fws") or [session()["fw"]]
    mons = (
        []
        if a.no_monitor
        else [
            SamplingMonitor(
                f, log=lambda *x, t=f.hw.tile: dh.log(f"[monitor tile {t}]", *x), exit_on_fail=not a.retry_on_timeout
            )
            for f in fws
        ]
    )
    allp = dh.load_prompts(home, 32)
    if B == 1:
        groups = [[allp[i]] for i in range(a.prompts)]
        cases = [
            (s, [dict(zip(("temperature", "top_k", "top_p"), SETTINGS[s]))], [a.seed]) for s in a.settings.split(",")
        ]
    else:
        from sampling_mix import user_params  # shared with Plan A batch 32: user u -> setting u % 4, seed 1234 + u

        groups = [allp[:32]]
        up = user_params(32)
        cases = [
            (
                "sampling_mix (greedy/T0.7k50p0.9/T1.0k0p1.0/T0.6k20p0.95 by u%4, seed 1234+u)",
                [dict(temperature=t, top_k=k, top_p=p) for (t, k, p, _) in up],
                [sd for (*_, sd) in up],
            )
        ]
    R, report = [], {"model": a.model, "batch": B, "steps": N, "cases": []}

    def say(s):
        R.append(s)
        dh.log(s)

    # 1) reference: host C library on the same device logits (Plan A shape), all cases first
    refs = {}
    for name, params, seeds in cases:
        for gi, prompts in enumerate(groups):
            g = h.run_group(gi, prompts, make_sampler("hostref", "full"), params, seeds, None)
            refs[(name, gi)] = [u["tokens"] for u in g["users"]]
    say(f"reference done ({len(refs)} groups)")
    neg_user = (a.neg_user if a.neg_user is not None else (1 if B > 1 else 0)) if a.negative_control else None
    inject = {"done": a.inject_trap is None}
    rst = {"next": a.restart_every, "n": 0, "ms": [], "levels": {}, "ctl": None, "last": -1}

    def poll_cb(done):
        if a.restart_every and done < rst["last"]:  # a new Plan B run (group) started: restart schedule from 0
            rst["next"] = a.restart_every
        rst["last"] = done
        if a.restart_every and done >= rst["next"]:
            if rst["ctl"] is None:
                rst["ctl"] = session()["fw"].ctl
            ctl = rst["ctl"]
            if rst["n"] % 2:
                ctl.rnmi(0xF, 2, timeout=0.5)  # RNMI park mode: L2 first, restart() then finds everything parked
            out = ctl.restart(None, warm=True)
            lv = "L2" if rst["n"] % 2 else "L1"
            rst["levels"][lv] = rst["levels"].get(lv, 0) + 1
            rst["ms"].append(out["t_total"] * 1e3)
            rst["n"] += 1
            rst["next"] = done + a.restart_every
        if not inject["done"] and done >= a.inject_at:
            inject["done"] = True
            st, rep = fws[a.inject_tile].ctl.inject(a.inject_trap)  # illegal instruction on that hart
            say(
                f"injected a trap on hart {a.inject_trap} at decode step {done} (mailbox status {st}); t={time.time():.3f}"
            )

    for m in mons:
        m.start()
    neg_ok = []
    # 2) Plan B on the firmware
    for name, params, seeds in cases:
        ident_users, total_users, tok_ok, tok_all, ms, enq, reads, tim, diags, freqs = 0, 0, 0, 0, [], 0, 0, [], [], []
        for gi, prompts in [(gi, p) for gi, p in enumerate(groups) for _ in range(a.repeat)]:
            x_seeds = None
            if neg_user is not None:
                x_seeds = list(seeds)
                x_seeds[neg_user] += 1
            if a.retry_on_timeout:
                from plan_b import run_group_planb_retry

                if rst["ctl"] is None:
                    rst["ctl"] = session()["fw"].ctl
                r = run_group_planb_retry(
                    h,
                    pb,
                    prompts,
                    params,
                    seeds,
                    N,
                    chunk=a.retry_on_timeout,
                    ctl=rst["ctl"] if len(fws) == 1 else None,
                    inject_at=a.inject_timeout_at,
                    inject_tile=a.inject_tile,
                    x280_seeds=x_seeds,
                    log=say,
                )
                say(f"  {len(r['retries'])} timeout(s) recovered in this group")
                for x in r["retries"]:
                    say(
                        f"  retry at step {x['step']}: recovery {x['recover_s'] * 1e3:.1f} ms (warm restart "
                        f"{x['restart_s'] * 1e3:.1f} ms, per tile {x.get('restart_tiles_ms')}), wait status "
                        f"{x['wait_status']}, firmware error {x['error']}"
                    )
            else:
                r = run_group_planb(
                    h, pb, prompts, params, seeds, N, poll=True, diag=not a.no_diag, x280_seeds=x_seeds, poll_cb=poll_cb
                )
            ref = refs[(name, gi)]
            diverged = []
            for b in range(B):
                total_users += 1
                same = r["tokens"][b] == ref[b][: N + 1]
                ident_users += same
                tok_all += N + 1
                tok_ok += sum(x == y for x, y in zip(r["tokens"][b], ref[b]))
                if not same:
                    bad = next(i for i, (x, y) in enumerate(zip(r["tokens"][b], ref[b])) if x != y)
                    diverged.append(b)
                    say(
                        f"  {'DIVERGES (expected)' if b == neg_user else 'MISMATCH'} {name} group {gi} user {b}: first differing "
                        f"token index {bad} (= decode step {bad - 1}; plan B {r['tokens'][b][bad]} ref {ref[b][bad]})"
                    )
            if neg_user is not None:
                neg_ok.append(diverged == [neg_user])
            ms.append(r["decode_s"] / N * 1e3)
            enq += r.get("enqueues", N)
            reads += r.get("arena_reads", 0)
            tim += r.get("timing", [])
            if "diag" in r:
                diags.append(r["diag"])
                d = r["diag"]
                span = (d[5] - d[4]) & 0xFFFFFFFF
                if r["decode_s"] < 2.5 and d[0] > 1:
                    freqs.append(span / (r["decode_s"] * (d[0] - 1) / d[0]) / 1e6)
        if a.restart_every:
            say(
                f"warm restarts during {name}: {rst['n']} ({rst['levels']}), restart ms median "
                f"{statistics.median(rst['ms']) if rst['ms'] else 0:.2f}, max {max(rst['ms']) if rst['ms'] else 0:.2f}; "
                f"wait_status 0x{session()['fw'].r32(__import__('plan_b').OFF_WAIT_STATUS):x}, seq_skips {session()['fw'].r32(__import__('l2cpu.sampling.layout', fromlist=['L2S_OFF_SEQ_SKIPS']).L2S_OFF_SEQ_SKIPS)}"
            )
        us = lambda c: c / 1750.0  # noqa: E731
        med = lambda k: statistics.median(t[k] for t in tim)  # noqa: E731
        split = (
            {k: round(us(med(k)), 1) for k in ("cyc_total", "cyc_read", "cyc_sample", "cyc_write", "cyc_wait")}
            if tim
            else {}
        )
        line = (
            f"{name}: identical {ident_users}/{total_users} users ({tok_ok}/{tok_all} tokens); Plan B ms/token median "
            f"{statistics.median(ms):.3f} (per group {[round(x, 3) for x in ms]}); host device ops during decode: "
            f"{enq} enqueues for {N * len(groups) * a.repeat} steps + {reads} arena polls; x280 per-step us (median) {split}"
        )
        if r.get("timing_tiles"):
            per_tile = [
                {
                    k: round(us(statistics.median(t[k] for t in recs)), 1)
                    for k in ("cyc_total", "cyc_read", "cyc_sample", "cyc_wait")
                }
                for recs in r["timing_tiles"]
            ]
            line += f"; x280 per tile (last group, us) {per_tile}"
        if diags:
            cnt = sum(d[0] for d in diags)
            line += (
                f"; device gap notify (doorbell) -> wait-release: mean {sum(d[1] for d in diags)/cnt/1350:.1f} us, max {max(d[2] for d in diags)/1350:.1f} us (1350 MHz ticks);"
                f" wait kernel entry -> release mean {sum(d[3] for d in diags)/cnt:.0f} ticks (n={cnt})"
                + (
                    f"; Tensix wall clock ~{statistics.median(freqs):.0f} MHz (from tick span vs host time)"
                    if freqs
                    else ""
                )
            )
        say(line)
        report["cases"].append(
            dict(
                name=name,
                identical_users=ident_users,
                users=total_users,
                tokens_ok=tok_ok,
                tokens=tok_all,
                ms_per_token=ms,
                x280_us=split,
                diag=diags,
                enqueues=enq,
            )
        )
    if mons:
        say(f"monitor: {[m.stop() for m in mons]}")
    rc = 0
    if neg_user is not None:
        rc = 0 if neg_ok and all(neg_ok) else 1
        say(
            f"NEGATIVE CONTROL {'PASS' if rc == 0 else 'FAIL'}: x280 seed of user {neg_user} + 1 -> exactly that user diverged in {sum(neg_ok)}/{len(neg_ok)} groups"
        )
    else:
        rc = 0 if all(c["identical_users"] == c["users"] for c in report["cases"]) else 1
    print("\n=== SUMMARY ===\n" + "\n".join(R), flush=True)
    out = a.out or os.path.join(deps.out_dir(), f"accept_plan_b_{a.model.split('/')[-1]}_b{B}.json")
    json.dump(report, open(out, "w"), indent=1)
    h.ttnn.close_mesh_device(h.mesh)
    return rc


if __name__ == "__main__":
    try:
        rc = main()
        sys.stdout.flush()
        os._exit(rc or 0)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
