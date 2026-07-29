#!/usr/bin/env python3
"""Fit an adoption gate for the mesh placement from the corpus A/B, using only COMPILE-TIME features.

Reads results_v2/corpus_mesh.jsonl (mask 0 vs mask 8192) and correlates the measured speedup with quantities
the program factory can compute without measuring anything: traffic volumes from the picked config, the DRAM
floor, etc. Prints candidate single-threshold gates ranked by how cleanly they separate wins from losses, and
what each would deliver on the corpus.

usage: fit_mesh_gate.py [--jsonl name]
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from regime_a_model import production_pick  # noqa: E402

JSONL = f"{HERE}/results_v2/corpus_mesh.jsonl"
if "--jsonl" in sys.argv:
    JSONL = f"{HERE}/results_v2/{sys.argv[sys.argv.index('--jsonl') + 1]}"
TB = 2048
PEAK = 512e9


def cd(v):
    return -(-v // 32)


def features(M, K, N):
    Pk, Ns, Sm, kb, nsb = production_pick(cd(M), cd(K), cd(N))[:5]
    Mt, Kt, Nt = cd(M), cd(K), cd(N)
    K_slice = -(-(-(-Kt // Pk)) // (kb * 8)) * (kb * 8)
    M_block = -(-Mt // Sm)
    N_band = -(-Nt // 8)
    N_own = -(-N_band // Ns)
    N_sub = nsb if nsb else N_own
    N_bpc = -(-N_own // N_sub)
    W = (K_slice // kb) // 8
    preaders = Pk * Ns * Sm
    ncores = 8 * preaders
    shard = W * M_block * kb * TB
    ring_b = preaders * 8 * 7 * shard          # every ring edge carries 7 shards
    in1_b = K * N * 2
    in0_b = ncores * shard                     # own-shard DRAM reads
    out_b = M * N * 2
    red_b = (Pk - 1) * Ns * Sm * 8 * N_bpc * M_block * N_sub * TB
    floor_us = (in0_b + in1_b + out_b) / PEAK * 1e6
    return dict(Pk=Pk, Ns=Ns, Sm=Sm, kb=kb, nsb=nsb, Mt=Mt, preaders=preaders, shard=shard,
                ring_b=ring_b, in1_b=in1_b, in0_b=in0_b, out_b=out_b, red_b=red_b, floor_us=floor_us,
                ring_over_in1=ring_b / in1_b, ring_over_dram=ring_b / (in0_b + in1_b + out_b),
                ring_plus_red_over_in1=(ring_b + red_b) / in1_b)


def main():
    rows = {}
    for line in open(JSONL):
        r = json.loads(line)
        if r.get("outcome") != "ok":
            continue
        b = r["modes"].get("0", {}).get("median_us")
        v = r["modes"].get("8192", {}).get("median_us")
        if not (b and v):
            continue
        rows.setdefault((r["M"], r["K"], r["N"]), []).append((b, v))
    data = []
    for (M, K, N), lst in rows.items():
        b = statistics.median([x[0] for x in lst])
        v = statistics.median([x[1] for x in lst])
        d = 100 * (b - v) / b
        f = features(M, K, N)
        f.update(M=M, K=K, N=N, base=b, mesh=v, delta=d, floor_pct=100 * f["floor_us"] / b)
        data.append(f)
    data.sort(key=lambda z: -z["delta"])
    print(f"{'shape':17s} {'cfg':16s} {'base':>7s} {'mesh':>7s} {'delta%':>7s} {'ring/in1':>9s} "
          f"{'(ring+red)/in1':>15s} {'ring/DRAM':>10s} {'%floor':>7s}")
    for f in data:
        name = "{}x{}x{}".format(f['M'], f['K'], f['N'])
        cfg = str((f['Pk'], f['Ns'], f['Sm'], f['kb'], f['nsb']))
        print(f"{name:17s} {cfg:16s} {f['base']:7.2f} {f['mesh']:7.2f} "
              f"{f['delta']:+7.2f} {f['ring_over_in1']:9.2f} {f['ring_plus_red_over_in1']:15.2f} "
              f"{f['ring_over_dram']:10.2f} {f['floor_pct']:6.0f}%")

    wins = [f for f in data if f["delta"] >= 2]
    loss = [f for f in data if f["delta"] <= -2]
    print(f"\n{len(data)} shapes: {len(wins)} win >=2%, {len(loss)} lose <=-2%, "
          f"{len(data)-len(wins)-len(loss)} neutral")

    # candidate single-feature thresholds; score = corpus-wide mean gain if we adopt only above the threshold
    print(f"\n{'feature':24s} {'thresh':>8s} {'adopt':>6s} {'mean gain all':>14s} {'worst adopted':>14s}")
    best = None
    for feat in ("ring_over_in1", "ring_plus_red_over_in1", "ring_over_dram"):
        vals = sorted({round(f[feat], 3) for f in data})
        for t in vals:
            adopted = [f for f in data if f[feat] >= t]
            gains = [(f["delta"] if f[feat] >= t else 0.0) for f in data]
            mean_gain = statistics.mean(gains)
            worst = min([f["delta"] for f in adopted], default=0.0)
            if best is None or (worst >= -2.0 and mean_gain > best[2]):
                if worst >= -2.0:
                    best = (feat, t, mean_gain, len(adopted), worst)
    for feat in ("ring_over_in1", "ring_plus_red_over_in1", "ring_over_dram"):
        vals = sorted({round(f[feat], 3) for f in data})
        rows_out = []
        for t in vals:
            adopted = [f for f in data if f[feat] >= t]
            if not adopted:
                continue
            mean_gain = statistics.mean([(f["delta"] if f[feat] >= t else 0.0) for f in data])
            worst = min(f["delta"] for f in adopted)
            rows_out.append((mean_gain, t, len(adopted), worst))
        rows_out.sort(reverse=True)
        for mg, t, n, w in rows_out[:3]:
            print(f"{feat:24s} {t:8.3f} {n:6d} {mg:+13.2f}% {w:+13.2f}%")
    if best:
        print(f"\nbest no-regression gate: {best[0]} >= {best[1]} -> adopt on {best[3]} shapes, "
              f"corpus mean gain {best[2]:+.2f}%, worst adopted {best[4]:+.2f}%")


main()
