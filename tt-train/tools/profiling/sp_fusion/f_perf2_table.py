#!/usr/bin/env python3
"""Tables from PERF2 lines of the SP fused-op tests. Usage: f_perf2_table.py <log> [<log> ...]"""
import re, sys

rows = []
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        if not line.startswith("PERF2 "):
            continue
        m = re.match(
            r"PERF2 \[(\w+)\] (\w+) B=(\d+).*?links=(\d).*?(bf16acc|fp32acc).*?payload=(\w+) pc\(per_core_N=(\d+) sub=(\dx\d)\): fused ([\d.]+) \| unfused ([\d.]+) \(([\d.]+)x\) \| linear ([\d.]+) \| mm_sp_alone ([\d.]+) \| (ag|rs)_alone ([\d.]+) \| exposed_\w+ (-?[\d.]+)(.*)",
            line,
        )
        if not m:
            print("UNPARSED:", line[:120])
            continue
        topo, name, B, links, ckc, payload, pcn, sub, fused, unf, sp, lin, mm, kind, ccl, exp, rest = m.groups()
        g = lambda pat: (re.search(pat, rest).group(1) if re.search(pat, rest) else "")
        rows.append(
            dict(
                topo=topo,
                name=name,
                B=int(B),
                ckc=ckc,
                payload=payload,
                pcn=pcn,
                sub=sub,
                fused=float(fused),
                unf=float(unf),
                sp=float(sp),
                lin=float(lin),
                mm=float(mm),
                kind=kind,
                ccl=float(ccl),
                exp=float(exp),
                pcc_ref=g(r"NUMERICS vs fp32 ref: .*?PCC: ([\d.]{1,7})"),
                maxd_ref=g(r"NUMERICS vs fp32 ref: .*?max\|d\|=([\d.e+-]{1,6})"),
                ulp_ref=g(r"NUMERICS vs fp32 ref: .*?ulp\(max/p99.9\)=([\d.]+/[\d.]+)"),
                pcc_b=g(r"bfp8-vs-bf16 fused: .*?PCC: ([\d.]{1,7})"),
                maxd_b=g(r"bfp8-vs-bf16 fused: .*?max\|d\|=([\d.e+-]{1,6})"),
                ulp_b=g(r"bfp8-vs-bf16 fused: .*?ulp\(max/p99.9\)=([\d.]+/[\d.]+)"),
                ulp_g=g(r"gathered bfp8 vs bf16 input ulp\(max/p99.9\)=([\d.]+/[\d.]+)"),
            )
        )
rows.sort(key=lambda r: (r["kind"], r["topo"], r["ckc"], r["payload"], r["B"], r["name"]))
print(
    f"{'op':3s} {'topo':4s} {'ckc':7s} {'payload':7s} {'shape':26s} {'cfg':8s} {'fused':>6s} {'unfused':>7s} {'x':>5s} {'linear':>6s} {'mm_sp':>6s} {'ccl':>5s} {'expo':>6s}  {'PCC ref':8s} {'max|d|':6s} {'ulp max/p99.9':13s} {'PCC vs bf16':11s} {'max|d|':6s} {'ulp max/p99.9':13s} {'gathered ulp':12s}"
)
for r in rows:
    print(
        f"{r['kind']:3s} {r['topo']:4s} {r['ckc']:7s} {r['payload']:7s} {r['name']:26s} {r['pcn']+'/'+r['sub']:8s} {r['fused']:6.0f} {r['unf']:7.0f} {r['sp']:5.2f} {r['lin']:6.0f} {r['mm']:6.0f} {r['ccl']:5.0f} {r['exp']:6.0f}  {r['pcc_ref']:8s} {r['maxd_ref']:6s} {r['ulp_ref']:13s} {r['pcc_b']:11s} {r['maxd_b']:6s} {r['ulp_b']:13s} {r['ulp_g']:12s}"
    )
