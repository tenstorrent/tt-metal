#!/usr/bin/env python3
"""standings.py [SUITE ...]: main, v3q and v4 per suite from stored result CSVs, row by row and mean (cycles/packet at
1.35 GHz; RESULTS §98 "Where v4 stands", V4_PERF_GAPS.md). Labels are cross-run: see V4_PERF_GAPS.md for each source."""
import csv, os, statistics, sys

R = "/home/aagarwal/tt-metal/fabric_exp/results"


def rows(f):
    out = {}
    p = f"{R}/{f}.csv"
    if not os.path.exists(p):
        return None
    for r in csv.DictReader(open(p)):
        if "avg_packets_per_s" in r:
            k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r["packet_size"])
            v = 1.35e9 / float(r["avg_packets_per_s"])
        else:  # vc1_tests format: one row per sender; keep the slowest sender per test
            k = (r["test_name"], "", "")
            v = 1.35e9 / float(r["packets_per_second"])
            if k in out:
                out[k] = max(out[k], v)
                continue
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = v
    return out


SUITES = [
    ("2D 4352 B", [("main", "mpost_v1_13129088_4352"), ("v3q", "v3q2d_13129088_4352"),
                   ("v4", "s1two_v4gi_13129088_4352")]),
    ("1D 4352 B", [("main", "mpost_v1_13129088_1d"), ("v3q w16", "v3qw16_13129088_1d"),
                   ("v3q w24", "v3qw24_13129088_1d"), ("v4", "s1two_v4gi_13129088_1d_m1")]),
    ("mix", [("main(a08)", "a8v1_13129088_mix"), ("v3q", "v3q_13129088_mix"), ("v4", "s1two_v4gi_13129088_mix")]),
    ("burst", [("v3q(a08)", "p34_v3zy_13129088_burst"), ("v4", "s1two_v4gi_13129088_burst")]),
    ("ring", [("main fw", "main1d_tier_ring"), ("main ka", "v1_13129088_ring"), ("v3q", "v3q_13129088_ring"),
              ("v4tm", "rsa_tm_13129088_ring"), ("v4rs", "rsa_rs_13129088_ring"), ("v4rr", "rsa_rr_13129088_ring")]),
    ("torusy", [("main ka", "v1_13129088_torusy"), ("v3q", "v3q_13129088_torusy"),
                ("v4tm", "rsa_tm_13129088_torusy"), ("v4rs", "rsa_rs_13129088_torusy"),
                ("v4rr", "rsa_rr_13129088_torusy")]),
    ("1-ERISC 2D", [("main", "se_fab1_v1_13129088_4352"), ("v4", "s1fab1_v4se_13129088_4352")]),
    ("1-ERISC 1D", [("main", "se_fab1_v1_13129088_1d"), ("v4", "s1fab1_v4se_13129088_1d_m1")]),
    ("1-ERISC ring", [("main", "rsea_v1_13129088_rse"), ("main b", "rseb_v1_13129088_rse"),
                      ("v4tm", "rsa_tm_13129088_rse"), ("v4rs", "rsa_rs_13129088_rse"),
                      ("v4rr", "rsa_rr_13129088_rse")]),
    ("mux", [("main", "mxa_v1_13129088_mux"), ("main b", "mxb_v1_13129088_mux"), ("v4tm", "rsa_tm_13129088_mux"),
             ("v4rs", "rsa_rs_13129088_mux"), ("v4rr", "rsa_rr_13129088_mux")]),
    ("VC1 split", [("main", "mpostvc1_v1_default_p1_auto_13129088_vc1_tests"),
                   ("v4", "mpostvc1_v4gi_default_p1_o3_13129088_vc1_tests")]),
]

only = sys.argv[1:]
for name, cols in SUITES:
    if only and name not in only:
        continue
    data = [(c, rows(f)) for c, f in cols]
    data = [(c, d) for c, d in data if d]
    ks = [k for k in data[0][1] if all(k in d for _, d in data[1:]) and "MixCmd" not in k[0]]
    print(f"\n## {name} ({len(ks)} rows)")
    print(f"{'row':52}" + "".join(f"{c:>10}" for c, _ in data))
    for k in ks:
        lab = f"{k[0]} {k[1]} {k[2]}" + (f" #{k[3]}" if k[3] else "")
        print(f"{lab[:52]:52}" + "".join(f"{d[k]:10.1f}" for _, d in data))
    print(f"{'MEAN':52}" + "".join(f"{statistics.mean(d[k] for k in ks):10.1f}" for _, d in data))
