# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize test_attention_perf ops CSV: per tag, per-op device time (max over chips, mean over iterations)
and SDPA FPU utilization vs the sdpa_perf_model ideal (4096 FMA-FLOP/cycle/core LoFi, x2 HiFi2, x4 HiFi4).

    python models/demos/mimo_v2_d_p/tests/perf/analyze_attention.py <csv> [--sp 2 --tp 2 --cores 120 --ghz 1.35 --fid 2]
"""

import argparse
import collections
import csv
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sp", type=int, default=2)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--cores", type=int, default=100, help="SDPA compute cores (11x10 grid minus the CCL column)")
    ap.add_argument("--ghz", type=float, default=1.35)
    ap.add_argument("--fid", type=float, default=2.0, help="fidelity multiplier (LoFi 1, HiFi2 2, HiFi4 4)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    # tag -> iteration -> device -> [(op, us)]
    data = collections.OrderedDict()
    cur, it = None, collections.Counter()
    for x in rows:
        if x["OP TYPE"] == "signpost":
            c = x["OP CODE"]
            if c.endswith("_start"):
                cur = c[: -len("_start")]
                it[cur] += 1
                data.setdefault(cur, collections.defaultdict(lambda: collections.defaultdict(list)))
            elif c.endswith("_end"):
                cur = None
            continue
        if cur is None:
            continue
        data[cur][it[cur]][int(x["DEVICE ID"])].append((x["OP CODE"], float(x["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3))
    summary = []
    for tag, iters in data.items():
        m = re.match(r"(\w+)_C(\d+)_ctx(\d+)", tag)
        kind, C, ctx = m.group(1), int(m.group(2)), int(m.group(3))
        per_op = collections.defaultdict(list)
        totals = []
        for i, devs in iters.items():
            n_ops = min(len(v) for v in devs.values())
            op_t = collections.defaultdict(float)
            for j in range(n_ops):
                name = next(iter(devs.values()))[j][0]
                op_t[f"{j:02d} {name}"] = max(v[j][1] for v in devs.values())
            for k, v in op_t.items():
                per_op[k].append(v)
            totals.append(sum(op_t.values()))
        total = sum(totals) / len(totals)
        print(f"\n=== {tag}: attention layer {total:.1f} us (sum of per-op max over chips) ===")
        sdpa_us = 0.0
        for k, v in per_op.items():
            t = sum(v) / len(v)
            if "SDPA" in k or "Sdpa" in k:
                sdpa_us += t
            print(f"  {k:<60s} {t:9.1f} us  {100 * t / total:5.1f}%")
        # SDPA ideal: per chip, nq_l heads, Q = C local rows attending causally to [0, pos]
        nq_l = 64 // a.tp
        DH, DV = 192, (192 if kind == "SWA" else 128)
        chunk = C * a.sp
        kv0 = ctx - chunk
        if kind == "SWA":
            pairs_total = chunk * 128  # ~window keys per query
        else:
            pairs_total = chunk * kv0 + chunk * (chunk + 1) / 2
        pairs_chip = pairs_total / a.sp
        flops = 2 * pairs_chip * (DH + DV) * nq_l
        ideal_us = flops / (a.cores * 4096) * a.fid / (a.ghz * 1e3)
        print(f"  SDPA {sdpa_us:.1f} us, ideal {ideal_us:.1f} us @ {a.cores} cores fid x{a.fid:g} -> FPU util {100 * ideal_us / max(sdpa_us, 1e-9):.1f}%"
              f"  ({flops / sdpa_us / 1e6:.1f} TFLOP/s/chip)")
        summary.append((tag, total, sdpa_us, 100 * ideal_us / max(sdpa_us, 1e-9)))
    print("\nSUMMARY tag layer_us sdpa_us util%")
    for t in summary:
        print(f"SUMMARY {t[0]:<24s} {t[1]:9.1f} {t[2]:9.1f} {t[3]:5.1f}")


if __name__ == "__main__":
    main()
