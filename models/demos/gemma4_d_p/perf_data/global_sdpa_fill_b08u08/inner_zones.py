"""Sum compute inner-loop zone time per core (TRISC_1 = math) for the SDPA op on the median device."""

import csv, sys, glob, collections, statistics as st

R = sys.argv[1]
rows = list(csv.DictReader(open(glob.glob(R + "/ops_perf_results_*.csv")[0])))
i0 = [i for i, r in enumerate(rows) if r["OP CODE"].endswith("-start")][-1]
win = {}
for r in rows[i0 + 1 :]:
    if r["OP CODE"] == "RingJointSDPADeviceOperation" and r["DEVICE ID"] not in win:
        win[r["DEVICE ID"]] = (
            int(r["DEVICE FW START CYCLE"]),
            int(r["DEVICE FW END CYCLE"]),
            float(r["DEVICE KERNEL DURATION [ns]"]) / 1e3,
        )
ks = sorted(win, key=lambda d: win[d][2])
d = ks[len(ks) // 2]
print("SDPA kernel us min/med/max %.1f/%.1f/%.1f, device %s" % (win[ks[0]][2], win[d][2], win[ks[-1]][2], d))
a, b, _ = win[d]
op = {}
tot = collections.defaultdict(lambda: collections.defaultdict(float))
cnt = collections.defaultdict(lambda: collections.defaultdict(int))
with open(glob.glob(R + "/profile_log_device.csv")[0]) as fh:
    next(fh)
    next(fh)
    for line in fh:
        p = line.split(",", 13)
        if p[0] != d:
            continue
        t = int(p[5])
        if t < a or t > b:
            continue
        risc, zn = p[3], p[10]
        if zn.endswith("-FW"):
            continue
        k = (int(p[1]), int(p[2]), risc, zn)
        if p[11] == "ZONE_START":
            op[k] = t
        elif p[11] == "ZONE_END" and k in op:
            tot[(risc, zn)][k[:2]] += (t - op.pop(k)) / 1350
            cnt[(risc, zn)][k[:2]] += 1
for risc, zn in sorted(tot, key=lambda x: (x[0], -st.median(tot[x].values()))):
    v = tot[(risc, zn)]
    c = cnt[(risc, zn)]
    print(
        f"{risc:8s} {zn:24s} cores {len(v):3d}  per-core total med {st.median(v.values()):8.1f} us  count med {st.median(c.values()):6.0f}  per-call {st.median(v.values())/max(1,st.median(c.values())):6.2f} us"
    )
