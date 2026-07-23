#!/usr/bin/env python3
"""Pilot: validate sweep_worker end-to-end and measure per-candidate SUBPROCESS wall (drives the campaign
runtime estimate). Runs a handful of (shape,config) candidates + one config=None baseline, times each
subprocess, and prints outcome/wall/pcc/subprocess-seconds."""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
WORKER = f"{HERE}/sweep_worker.py"
ITERS = 8

# (M,K,N, Pk,Ns,Sm,kb,nsb, do_pcc). Pk=0 => config=None baseline.
CANDS = [
    (256, 2048, 1024, 0, 0, 0, 0, 0, 1),  # baseline (prod pick)
    (256, 2048, 1024, 4, 1, 2, 2, 4, 1),  # table winner
    (256, 2048, 1024, 4, 1, 2, 2, 2, 1),  # neighbour
    (256, 2048, 1024, 1, 1, 1, 1, 1, 1),  # degenerate 8-core (valid-but-slow)
    (256, 2048, 1024, 6, 1, 1, 1, 1, 1),  # deep-K
    (32, 6144, 6144, 3, 1, 1, 4, 6, 1),  # wide-N Mt=1
]


def run(c):
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")
    args = [sys.executable, WORKER, *[str(x) for x in c[:8]], str(ITERS), str(c[8])]
    t = time.time()
    try:
        r = subprocess.run(args, env=env, cwd=ROOT, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "sweep_worker"], capture_output=True)
        return {"outcome": "hang"}, time.time() - t
    dt = time.time() - t
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    if line is None:
        return {"outcome": "runtime", "err": (r.stderr or r.stdout)[-300:]}, dt
    return json.loads(line), dt


def main():
    print(f"{'shape':18s} {'config':22s} {'outcome':10s} {'wall_us':>9s} {'pcc':>9s} {'proc_s':>7s}", flush=True)
    for c in CANDS:
        res, dt = run(c)
        shape = f"{c[0]}x{c[1]}x{c[2]}"
        cfg = "None(prod)" if c[3] == 0 else f"{c[3]},{c[4]},{c[5]},{c[6]},{c[7]}"
        print(
            f"{shape:18s} {cfg:22s} {res.get('outcome',''):10s} "
            f"{str(res.get('wall_us')):>9s} {str(res.get('pcc')):>9s} {dt:>7.1f}",
            flush=True,
        )
        if res.get("err"):
            print(f"    err: {res['err'][:200]}", flush=True)


if __name__ == "__main__":
    main()
