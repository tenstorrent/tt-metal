#!/usr/bin/env python3
"""A/B the in1-optimal placement (diag bit12) with in0 + reduction ABLATED, so only in1 read, compute and
output write are exposed. Also reports the full-op numbers for context, and the DRAM utilisation.

Modes:
  0     full op, production placement
  4096  full op, in1-optimal (CROSS) placement
  21    = 1|4|16  in0 read + in0 ring forward + reduction skipped, production placement
  4117  = 21|4096 same ablation, in1-optimal placement

DRAM byte accounting (this is why the util column is trustworthy):
  full op : in0_read = ncores * shard_bytes, in1 = K*N*2, out = M*N*2
  mask 21 : in0_read = 0 (skipped). SKIP_REDUCTION makes EVERY split-K band write its own partial to the
            same output pages, so out = Pk * M*N*2, not M*N*2. Ignoring that would overstate the util.

usage: in1_isolated_ab.py [--relaunches N]
"""
import json, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
WORKER = f"{HERE}/corpus_ab_worker.py"
sys.path.insert(0, HERE)
from regime_a_model import production_pick  # noqa: E402

SHAPES = [
    (512, 6144, 2304), (512, 6144, 4608), (256, 15360, 768), (256, 6144, 4608),
    (32, 6144, 1536), (256, 2048, 2048), (256, 2048, 6144),
]
MASKS = [0, 4096, 21, 4117]
RELAUNCHES = 2
if "--relaunches" in sys.argv:
    RELAUNCHES = int(sys.argv[sys.argv.index("--relaunches") + 1])
PEAK_GBPS = 512.0


def cd(v):
    return -(-v // 32)


def geo(M, K, N):
    Pk, Ns, Sm, kb, nsb = production_pick(cd(M), cd(K), cd(N))[:5]
    Mt, Kt, Nt = cd(M), cd(K), cd(N)
    K_slice = -(-(-(-Kt // Pk)) // (kb * 8)) * (kb * 8)
    M_block = -(-Mt // Sm)
    W = (K_slice // kb) // 8
    ncores = 8 * Pk * Ns * Sm
    return dict(Pk=Pk, Ns=Ns, Sm=Sm, kb=kb, nsb=nsb, shard=W * M_block * kb * 2048, ncores=ncores)


def run(M, K, N, masks):
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")
    p = subprocess.run(
        [sys.executable, WORKER, str(M), str(K), str(N), ",".join(str(m) for m in masks), "0"],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=900)
    for line in p.stdout.splitlines():
        if line.startswith('{"outcome"'):
            return json.loads(line)
    return {"outcome": "runtime", "err": (p.stderr or p.stdout)[-200:]}


def main():
    print(f"{'shape':17s} {'cfg(Pk,Ns,Sm,kb,nsb)':22s} | {'FULL OP: prod':>13s} {'in1opt':>9s} {'delta':>7s} "
          f"| {'ISOLATED in1+compute+out: prod':>30s} {'in1opt':>9s} {'delta':>7s} | {'DRAM util prod':>14s} "
          f"{'in1opt':>8s}")
    for (M, K, N) in SHAPES:
        g = geo(M, K, N)
        acc = {m: [] for m in MASKS}
        for rl in range(RELAUNCHES):
            r = run(M, K, N, MASKS if rl % 2 == 0 else MASKS[::-1])
            if r.get("outcome") != "ok":
                print(f"{f'{M}x{K}x{N}':17s} FAIL {r.get('err','')[:80]}")
                break
            for m in MASKS:
                if str(m) in r["modes"]:
                    acc[m].append(r["modes"][str(m)]["median_us"])
        if not all(acc[m] for m in MASKS):
            continue
        med = {m: statistics.median(acc[m]) for m in MASKS}
        in1_b = K * N * 2
        out_b = M * N * 2
        in0_b = g["ncores"] * g["shard"]
        full_b = in0_b + in1_b + out_b
        iso_b = in1_b + g["Pk"] * out_b  # SKIP_REDUCTION => every band writes its partial
        def util(bytes_, us):
            return 100.0 * (bytes_ / 1e9) / (us / 1e6) / PEAK_GBPS
        d_full = 100 * (med[0] - med[4096]) / med[0]
        d_iso = 100 * (med[21] - med[4117]) / med[21]
        print(f"{f'{M}x{K}x{N}':17s} {str(tuple(g[k] for k in ('Pk','Ns','Sm','kb','nsb'))):22s} | "
              f"{med[0]:13.2f} {med[4096]:9.2f} {d_full:+6.1f}% | {med[21]:30.2f} {med[4117]:9.2f} "
              f"{d_iso:+6.1f}% | {util(iso_b, med[21]):13.0f}% {util(iso_b, med[4117]):7.0f}%")
    print(f"\nDRAM util = (in1 read + Pk x output write) / wall, vs {PEAK_GBPS:.0f} GB/s, for the ISOLATED mode.")
    print("FULL OP columns are the same placements with nothing ablated (context; not the isolation test).")


main()
