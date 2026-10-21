"""Per signposted layer: median-over-devices device kernel time of the SDPA op and of the whole layer (sum of op medians)."""

import csv, sys, glob, collections, statistics as st

rows = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/ops_perf_results_*.csv")[0])))
seg = None
data = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r["OP TYPE"] == "signpost":
        seg = r["OP CODE"][:-6] if r["OP CODE"].endswith("-start") else None
        continue
    if seg:
        data[seg][r["DEVICE ID"]].append(r)
for seg, devs in data.items():
    n = min(len(v) for v in devs.values())
    tot = 0
    sd = None
    for i in range(n):
        kd = [float(devs[d][i]["DEVICE KERNEL DURATION [ns]"]) / 1e3 for d in devs]
        m = st.median(kd)
        tot += m
        if devs[next(iter(devs))][i]["OP CODE"].startswith("RingJointSDPA"):
            sd = (min(kd), m, max(kd))
    print(f"{seg}: layer sum {tot:.1f} us over {n} ops; SDPA min/med/max {sd[0]:.1f}/{sd[1]:.1f}/{sd[2]:.1f} us")
