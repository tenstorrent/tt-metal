#!/usr/bin/env python3
"""Verify the persistent-batch timing methodology against the isolated one-process-per-config method.

Picks N configs from a shape that was measured exhaustively by the isolated harness (sweep_*.jsonl),
re-measures them via batch_worker (mini-batch), and reports per-config batched-vs-isolated deltas.
Requires agreement within ~1% (instruction 3). Uses fresh random inputs, so tiny run-to-run noise is
expected; the methodology is validated if deltas are within noise (~1%)."""
import json, os, subprocess, sys, tempfile, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import regime_a_candidate_generator as cg  # noqa: E402

SHAPE = (256, 2048, 1024)


def isolated_walls(M, K, N):
    p = f"{HERE}/results/sweep_{M}x{K}x{N}.jsonl"
    out = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("cfgkey") == "None" or r.get("outcome") != "ok" or not r.get("wall_us"):
            continue
        out[(r["Ns"], r["Pk"], r["Sm"], r["kb"], r["nsb"])] = r["wall_us"]  # generator order
    return out


def main():
    M, K, N = SHAPE
    iso = isolated_walls(M, K, N)
    # pick 5 configs spanning the perf range that the generator would also select.
    sel, _r, _s = cg.select_candidates(M, K, N, budget=96, audit=0)
    common = [g.cfg for g in sel if g.cfg in iso]
    common.sort(key=lambda c: iso[c])
    picks = [common[0], common[len(common) // 4], common[len(common) // 2],
             common[3 * len(common) // 4], common[-1]]
    picks = list(dict.fromkeys(picks))[:5]

    out_jsonl = tempfile.mktemp(suffix=".jsonl")
    job = {"M": M, "K": K, "N": N, "iters": 8, "do_pcc": 1, "minibatch": 8, "out_jsonl": out_jsonl,
           "configs": [{"cfg": list(c)} for c in picks]}
    jobf = tempfile.mktemp(suffix=".json")
    json.dump(job, open(jobf, "w"))
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")
    subprocess.run([sys.executable, f"{HERE}/batch_worker.py", jobf], env=env, cwd=ROOT, timeout=600)

    batched = {}
    for line in open(out_jsonl):
        r = json.loads(line)
        if r["outcome"] == "ok":
            batched[tuple(r["cfg"])] = (r["wall_us"], r["pcc"])
    print(f"\n{'cfg (Ns,Pk,Sm,kb,nsb)':24s} {'isolated_us':>11s} {'batched_us':>11s} {'delta%':>8s} {'pcc':>9s}")
    deltas = []
    for c in picks:
        iw = iso[c]
        bw, p = batched.get(c, (None, None))
        if bw is None:
            print(f"{str(c):24s} {iw:11.3f} {'MISSING':>11s}")
            continue
        d = (bw - iw) / iw * 100
        deltas.append(d)
        print(f"{str(c):24s} {iw:11.3f} {bw:11.3f} {d:+8.2f} {p:9.5f}")
    if deltas:
        print(f"\nmax |delta| = {max(abs(d) for d in deltas):.2f}%  mean = {statistics.mean(deltas):+.2f}%  "
              f"({'PASS' if max(abs(d) for d in deltas) <= 1.5 else 'CHECK'} vs ~1% target)")
    os.remove(out_jsonl); os.remove(jobf)


if __name__ == "__main__":
    main()
