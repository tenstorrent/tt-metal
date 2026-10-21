"""Global ring SDPA: per ring iteration, median over compute cores of RING-WAIT and C-KLOOP (us from op start)."""

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
print("SDPA kernel us min/med/max %.1f/%.1f/%.1f" % (win[ks[0]][2], win[ks[len(ks) // 2]][2], win[ks[-1]][2]))
pick = sys.argv[2].split(",") if len(sys.argv) > 2 else [ks[0], ks[len(ks) // 2], ks[-1]]
ev = collections.defaultdict(list)
op = {}
with open(glob.glob(R + "/profile_log_device.csv")[0]) as fh:
    next(fh)
    next(fh)
    for line in fh:
        p = line.split(",", 13)
        d = p[0]
        if d not in pick:
            continue
        t = int(p[5])
        a, b, _ = win[d]
        if t < a or t > b:
            continue
        zn = p[10]
        if zn not in ("RING-WAIT", "C-KLOOP", "TRISC-KERNEL", "NCRISC-KERNEL", "BRISC-KERNEL"):
            continue
        k = (d, int(p[1]), int(p[2]), p[3], zn)
        us = (t - a) / 1350
        if p[11] == "ZONE_START":
            op[k] = us
        elif p[11] == "ZONE_END" and k in op:
            ev[(d, (int(p[1]), int(p[2])), zn)].append((op.pop(k), us))
for d in pick:
    print(f"\n== device {d}: op {win[d][2]:.1f} us")
    rw = [v for (dd, c, z), v in ev.items() if dd == d and z == "RING-WAIT"]
    kl = [v for (dd, c, z), v in ev.items() if dd == d and z == "C-KLOOP"]
    print(
        f"  cores with RING-WAIT {len(rw)}, with C-KLOOP {len(kl)}; iterations per core {st.median(len(v) for v in rw) if rw else 0}"
    )
    n = max(len(v) for v in rw)
    tw = tc = 0
    for i in range(n):
        w = [v[i] for v in rw if len(v) > i]
        c = [v[i] for v in kl if len(v) > i]
        ws = st.median(x[0] for x in w)
        we = st.median(x[1] for x in w)
        wmax = max(x[1] for x in w)
        line = f"  iter {i}: wait {ws:7.1f}->{we:7.1f} (dur med {st.median(x[1]-x[0] for x in w):6.1f}, max end {wmax:7.1f})"
        if c:
            cs = st.median(x[0] for x in c)
            ce = st.median(x[1] for x in c)
            line += f" | compute {cs:7.1f}->{ce:7.1f} (dur med {st.median(x[1]-x[0] for x in c):6.1f}, max end {max(x[1] for x in c):7.1f})"
            tc += st.median(x[1] - x[0] for x in c)
        tw += st.median(x[1] - x[0] for x in w)
        print(line)
    print(f"  sum of median waits {tw:.1f} us, sum of median compute {tc:.1f} us")
