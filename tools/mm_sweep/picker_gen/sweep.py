#!/usr/bin/env python3
"""Resumable, hang-safe exhaustive sweep orchestrator for the picker-generalization campaign.

Enumerates every PLANNER-feasible config (via regime_a_model, nsb-lattice) for each swept shape and
measures it on-device through sweep_worker (one subprocess/relaunch per candidate). Infeasible configs
are rejected offline by the planner mirror and never launched. Results stream to per-shape JSONL with
atomic append; a restart skips already-measured (shape,config). Hangs (subprocess timeout) are classified
and trigger a device reset (tt-smi -r) before continuing.

Phases:
  initial : sweep all lattice configs, do_pcc=0 (wall only). Also measures config=None (prod baseline).
  rerun   : for each shape, remeasure the winner + every candidate within 5% + top-10, each with 3 fresh
            consecutive relaunches, do_pcc=1 (verify PCC). NON-interleaved: a candidate's 3 relaunches are
            independent fresh processes measured consecutively (no cross-candidate cycling).

Usage:
  python3 sweep.py initial [--split train,val,holdout] [--shape MxKxN ...] [--timeout 180]
  python3 sweep.py rerun   [--split ...] [--within 5.0] [--topk 10] [--relaunch 3]
  python3 sweep.py status
Reproduce/resume: re-run the SAME command; completed measurements are skipped automatically.
"""
import argparse, json, os, subprocess, sys, time, statistics, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
WORKER = f"{HERE}/sweep_worker.py"
RESULTS = f"{HERE}/results"
sys.path.insert(0, HERE)
import regime_a_model as model  # noqa: E402
import corpus as corpus_mod  # noqa: E402

ITERS = 8
os.makedirs(RESULTS, exist_ok=True)


def shape_key(M, K, N):
    return f"{M}x{K}x{N}"


def jsonl_path(M, K, N):
    return f"{RESULTS}/sweep_{shape_key(M,K,N)}.jsonl"


def cfg_key(c):
    return f"{c[0]},{c[1]},{c[2]},{c[3]},{c[4]}"  # Pk,Ns,Sm,kb,nsb ; baseline = "None"


def load_done(M, K, N, tag_filter=None):
    """Return dict cfgkey(+tag) -> list of records already measured (for resume / rerun aggregation)."""
    p = jsonl_path(M, K, N)
    done = {}
    if not os.path.exists(p):
        return done
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if tag_filter is not None and r.get("tag") != tag_filter:
            continue
        done.setdefault(r["cfgkey"], []).append(r)
    return done


def append_result(M, K, N, rec):
    """Atomic append of one measurement record (never clobbers prior results)."""
    with open(jsonl_path(M, K, N), "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def device_reset():
    print("  [reset] tt-smi -r after hang ...", flush=True)
    subprocess.run(["pkill", "-9", "-f", "sweep_worker"], capture_output=True)
    time.sleep(2)
    subprocess.run(["tt-smi", "-r"], capture_output=True, timeout=180)
    time.sleep(10)


def measure_once(M, K, N, cfg, do_pcc, timeout):
    """cfg = (Pk,Ns,Sm,kb,nsb) or None (baseline). Returns a result dict; outcome 'hang' on timeout."""
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")
    if cfg is None:
        argv = [str(M), str(K), str(N), "0", "0", "0", "0", "0"]  # Pk=0 sentinel -> config=None
    else:
        argv = [str(M), str(K), str(N), *[str(x) for x in cfg]]
    args = [sys.executable, WORKER, *argv, str(ITERS), "1" if do_pcc else "0"]
    try:
        r = subprocess.run(args, env=env, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"outcome": "hang", "wall_us": None, "samples": None, "pcc": None, "err": "timeout"}
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    if line is None:
        return {
            "outcome": "runtime",
            "wall_us": None,
            "samples": None,
            "pcc": None,
            "err": (r.stderr or r.stdout)[-300:],
        }
    return json.loads(line)


def swept_shapes(split_filter, shape_filter):
    out = []
    for sp, lst in corpus_mod.SWEPT.items():
        if split_filter and sp not in split_filter:
            continue
        for M, K, N in lst:
            if shape_filter and shape_key(M, K, N) not in shape_filter:
                continue
            out.append((M, K, N, sp))
    return out


# ------------------------------------------------------------------------------------------------
# Phase: initial
# ------------------------------------------------------------------------------------------------
def phase_initial(args):
    shapes = swept_shapes(args.split, args.shape)
    hang_streak = 0
    for M, K, N, sp in shapes:
        Mt, Kt, Nt = M // 32, K // 32, N // 32
        cfgs = [
            (r["Pk"], r["Ns"], r["Sm"], r["kb"], r["nsb"])
            for r in model.enumerate_feasible(Mt, Kt, Nt, nsb_mode="lattice")
        ]
        done = load_done(M, K, N, tag_filter="initial")
        todo = [None] + cfgs  # baseline first, then all configs
        remaining = [c for c in todo if (("None" if c is None else cfg_key(c)) not in done)]
        print(
            f"[{sp}] {shape_key(M,K,N)} Mt{Mt}Kt{Kt}Nt{Nt}: {len(cfgs)} configs, "
            f"{len(done)} done, {len(remaining)} to go",
            flush=True,
        )
        for i, cfg in enumerate(remaining):
            res = measure_once(M, K, N, cfg, do_pcc=False, timeout=args.timeout)
            rec = {
                "tag": "initial",
                "cfgkey": "None" if cfg is None else cfg_key(cfg),
                "Pk": 0 if cfg is None else cfg[0],
                "Ns": 0 if cfg is None else cfg[1],
                "Sm": 0 if cfg is None else cfg[2],
                "kb": 0 if cfg is None else cfg[3],
                "nsb": 0 if cfg is None else cfg[4],
                **res,
            }
            append_result(M, K, N, rec)
            if res["outcome"] == "hang":
                hang_streak += 1
                print(f"  HANG on {rec['cfgkey']} (streak {hang_streak}); resetting", flush=True)
                device_reset()
                if hang_streak >= 5:
                    print("  too many consecutive hangs; backing off 120s", flush=True)
                    time.sleep(120)
                    hang_streak = 0
            else:
                hang_streak = 0
            if (i + 1) % 50 == 0:
                oks = sum(1 for v in load_done(M, K, N, "initial").values() for r in v if r["outcome"] == "ok")
                print(f"  ...{i+1}/{len(remaining)} done ({oks} ok so far)", flush=True)
    print("initial phase complete", flush=True)


# ------------------------------------------------------------------------------------------------
# Phase: rerun (stability + PCC on winner + near-winners)
# ------------------------------------------------------------------------------------------------
def phase_rerun(args):
    shapes = swept_shapes(args.split, args.shape)
    for M, K, N, sp in shapes:
        init = load_done(M, K, N, "initial")
        oks = [
            (k, statistics.median([s for r in v for s in (r["samples"] or [])] or [r["wall_us"]]))
            for k, v in init.items()
            if any(r["outcome"] == "ok" and r["wall_us"] for r in v)
        ]
        oks = [(k, w) for (k, w) in oks if w]
        if not oks:
            print(f"[{sp}] {shape_key(M,K,N)}: no ok initial results; skip rerun", flush=True)
            continue
        oks.sort(key=lambda z: z[1])
        best = oks[0][1]
        # winner + everything within `within`% + top-k
        near = {k for (k, w) in oks if (w - best) / best * 100.0 <= args.within}
        near |= {k for (k, w) in oks[: args.topk]}
        near.add("None")  # always re-verify the production baseline
        print(
            f"[{sp}] {shape_key(M,K,N)}: best={best:.2f}us; reverifying {len(near)} candidates "
            f"x{args.relaunch} relaunches (PCC on)",
            flush=True,
        )
        donererun = load_done(M, K, N, "rerun")
        for k in sorted(near):
            have = len(donererun.get(k, []))
            cfg = None if k == "None" else tuple(int(x) for x in k.split(","))
            for j in range(have, args.relaunch):  # consecutive fresh relaunches (non-interleaved)
                res = measure_once(M, K, N, cfg, do_pcc=True, timeout=args.timeout)
                rec = {
                    "tag": "rerun",
                    "cfgkey": k,
                    "relaunch": j,
                    "Pk": 0 if cfg is None else cfg[0],
                    "Ns": 0 if cfg is None else cfg[1],
                    "Sm": 0 if cfg is None else cfg[2],
                    "kb": 0 if cfg is None else cfg[3],
                    "nsb": 0 if cfg is None else cfg[4],
                    **res,
                }
                append_result(M, K, N, rec)
                if res["outcome"] == "hang":
                    print(f"  HANG on rerun {k}; resetting", flush=True)
                    device_reset()
    print("rerun phase complete", flush=True)


def phase_status(args):
    for M, K, N, sp in swept_shapes(args.split, args.shape):
        Mt, Kt, Nt = M // 32, K // 32, N // 32
        ncfg = len(model.enumerate_feasible(Mt, Kt, Nt, nsb_mode="lattice"))
        init = load_done(M, K, N, "initial")
        by = {}
        for v in init.values():
            for r in v:
                by[r["outcome"]] = by.get(r["outcome"], 0) + 1
        rer = load_done(M, K, N, "rerun")
        print(
            f"[{sp}] {shape_key(M,K,N)}: {len(init)}/{ncfg+1} measured  outcomes={by}  rerun={len(rer)} cfgs",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["initial", "rerun", "status"])
    ap.add_argument("--split", type=lambda s: s.split(","), default=None)
    ap.add_argument("--shape", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--within", type=float, default=5.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--relaunch", type=int, default=3)
    args = ap.parse_args()
    {"initial": phase_initial, "rerun": phase_rerun, "status": phase_status}[args.phase](args)


if __name__ == "__main__":
    main()
