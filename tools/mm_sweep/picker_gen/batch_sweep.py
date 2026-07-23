#!/usr/bin/env python3
"""Generator-driven, persistent-batch sweep supervisor.

For each corpus_v2 shape: generate candidates via the theory-guided generator (budget=96, audit=8,
always including the real production-picker config), write a job file carrying each candidate's geometry
/ model-costs / reasons, and launch batch_worker to measure them. A PROGRESS-based watchdog monitors the
per-shape output JSONL; if no new record appears for `stall` seconds the worker is killed, the device is
reset (tt-smi -r), and the worker is relaunched (it resumes from the first missing config). Fully
resumable: shapes with all candidates already measured are skipped; re-running the same command continues.

Usage:
  python3 batch_sweep.py initial [--split train,val,holdout] [--shape MxKxN ...] [--stall 150]
  python3 batch_sweep.py rerun   [--within 5.0] [--topk 5] [--relaunch 3]   # sequential fresh relaunches
  python3 batch_sweep.py status
Reproduce/resume: re-run the SAME command.
"""
import argparse, json, os, subprocess, sys, time, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
RESULTS = f"{HERE}/results_v2"
JOBS = f"{HERE}/jobs"
WORKER = f"{HERE}/batch_worker.py"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import regime_a_model as model  # noqa: E402
import regime_a_candidate_generator as cg  # noqa: E402

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(JOBS, exist_ok=True)
ITERS = 8
MINIBATCH = 12


def out_path(M, K, N):
    return f"{RESULTS}/v2_{M}x{K}x{N}.jsonl"


def prod_pick_gen_order(Mt, Kt, Nt):
    try:
        pk = model.production_pick(Mt, Kt, Nt)
    except RuntimeError:
        return None
    return (pk[1], pk[0], pk[2], pk[3], pk[4])  # (Ns,Pk,Sm,kb,nsb)


def candidates_for(M, K, N):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    inc = prod_pick_gen_order(Mt, Kt, Nt)
    doc = cg.result_document(M, K, N, budget=96, audit=8, include=[inc] if inc else [])
    return doc["selected"], inc


def n_done(M, K, N):
    p = out_path(M, K, N)
    if not os.path.exists(p):
        return 0
    return sum(1 for line in open(p) if line.strip())


def device_reset():
    subprocess.run(["pkill", "-9", "-f", "batch_worker.py"], capture_output=True)
    time.sleep(2)
    subprocess.run(["tt-smi", "-r"], capture_output=True, timeout=180)
    time.sleep(10)


def run_shape(M, K, N, do_pcc, stall):
    cands, inc = candidates_for(M, K, N)
    if not cands:
        print(f"  {M}x{K}x{N}: no candidates; skip", flush=True)
        return
    expected = len(cands)
    op = out_path(M, K, N)
    job = {"M": M, "K": K, "N": N, "iters": ITERS, "do_pcc": int(do_pcc), "minibatch": MINIBATCH,
           "out_jsonl": op, "configs": cands}
    jobf = f"{JOBS}/job_{M}x{K}x{N}.json"
    json.dump(job, open(jobf, "w"))
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")

    attempts = 0
    while n_done(M, K, N) < expected and attempts < 12:
        attempts += 1
        proc = subprocess.Popen([sys.executable, WORKER, jobf], env=env, cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        last = n_done(M, K, N)
        last_t = time.time()
        while proc.poll() is None:
            time.sleep(5)
            cur = n_done(M, K, N)
            if cur > last:
                last, last_t = cur, time.time()
            elif time.time() - last_t > stall:
                print(f"  {M}x{K}x{N}: STALL at {cur}/{expected}; kill+reset (attempt {attempts})", flush=True)
                proc.kill()
                device_reset()
                break
        else:
            proc.wait()
    print(f"  {M}x{K}x{N}: {n_done(M,K,N)}/{expected} done (prod_pick {inc}, {attempts} launches)", flush=True)


def load_manifest():
    return json.load(open(f"{HERE}/corpus_v2_manifest.json"))["shapes"]


def phase_initial(args):
    shapes = load_manifest()
    if args.split:
        shapes = [s for s in shapes if s["split"] in args.split]
    if args.shape:
        shapes = [s for s in shapes if f"{s['M']}x{s['K']}x{s['N']}" in args.shape]
    # train first, then val, holdout
    order = {"train": 0, "val": 1, "holdout": 2}
    shapes.sort(key=lambda s: (order.get(s["split"], 9), s["M"], s["K"], s["N"]))
    t0 = time.time()
    for i, s in enumerate(shapes):
        M, K, N = s["M"], s["K"], s["N"]
        exp = s["n_candidates"]
        if n_done(M, K, N) >= exp:
            continue
        print(f"[{i+1}/{len(shapes)}] [{s['split']}] {M}x{K}x{N} ({exp} cands) "
              f"elapsed {(time.time()-t0)/3600:.1f}h", flush=True)
        run_shape(M, K, N, do_pcc=True, stall=args.stall)
    print("initial phase complete", flush=True)


def _median_all_samples(recs):
    s = [x for r in recs if r["outcome"] == "ok" for x in (r.get("samples") or [])]
    return statistics.median(s) if s else None


def phase_rerun(args):
    shapes = load_manifest()
    if args.split:
        shapes = [s for s in shapes if s["split"] in args.split]
    for s in shapes:
        M, K, N = s["M"], s["K"], s["N"]
        op = out_path(M, K, N)
        if not os.path.exists(op):
            continue
        recs = {}
        for line in open(op):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("tag") == "rerun":
                continue
            recs.setdefault(tuple(r["cfg"]), []).append(r)
        walls = {c: _median_all_samples(v) for c, v in recs.items()}
        walls = {c: w for c, w in walls.items() if w}
        if not walls:
            continue
        best = min(walls.values())
        near = {c for c, w in walls.items() if (w - best) / best * 100 <= args.within}
        near |= set(sorted(walls, key=walls.get)[: args.topk])
        # sequential fresh relaunches per candidate (NOT interleaved): each relaunch is its own worker.
        already = {}
        for line in open(op):
            r = json.loads(line) if line.strip() else {}
            if r.get("tag") == "rerun":
                already[tuple(r["cfg"])] = already.get(tuple(r["cfg"]), 0) + 1
        print(f"[rerun] {M}x{K}x{N}: {len(near)} near-winners x{args.relaunch}", flush=True)
        for c in sorted(near):
            for j in range(already.get(c, 0), args.relaunch):
                job = {"M": M, "K": K, "N": N, "iters": ITERS, "do_pcc": 1, "minibatch": 1,
                       "out_jsonl": op + ".rerun", "configs": [{"cfg": list(c)}]}
                jobf = f"{JOBS}/rerun_{M}x{K}x{N}.json"
                json.dump(job, open(jobf, "w"))
                # fresh worker per relaunch (own device open); tag + append into main file
                try:
                    os.remove(op + ".rerun")
                except OSError:
                    pass
                env = dict(os.environ)
                env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")
                subprocess.run([sys.executable, WORKER, jobf], env=env, cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
                if os.path.exists(op + ".rerun"):
                    for line in open(op + ".rerun"):
                        if line.strip():
                            r = json.loads(line); r["tag"] = "rerun"; r["relaunch"] = j
                            with open(op, "a") as f:
                                f.write(json.dumps(r) + "\n"); f.flush(); os.fsync(f.fileno())
    print("rerun phase complete", flush=True)


def phase_status(args):
    shapes = load_manifest()
    tot_done = tot = 0
    by_split = {}
    for s in shapes:
        M, K, N = s["M"], s["K"], s["N"]
        d, e = n_done(M, K, N), s["n_candidates"]
        tot_done += min(d, e); tot += e
        by_split.setdefault(s["split"], [0, 0])
        by_split[s["split"]][0] += min(d, e); by_split[s["split"]][1] += e
    print(f"total {tot_done}/{tot} = {tot_done/tot*100:.1f}%")
    for sp, (d, e) in by_split.items():
        print(f"  {sp:8s} {d}/{e}")
    nshapes_done = sum(1 for s in shapes if n_done(s["M"], s["K"], s["N"]) >= s["n_candidates"])
    print(f"shapes complete: {nshapes_done}/{len(shapes)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["initial", "rerun", "status"])
    ap.add_argument("--split", type=lambda s: s.split(","), default=None)
    ap.add_argument("--shape", nargs="*", default=None)
    ap.add_argument("--stall", type=int, default=150)
    ap.add_argument("--within", type=float, default=5.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--relaunch", type=int, default=3)
    args = ap.parse_args()
    {"initial": phase_initial, "rerun": phase_rerun, "status": phase_status}[args.phase](args)


if __name__ == "__main__":
    main()
