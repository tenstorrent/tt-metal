"""Global SDPA reader zones split into injector cores (have R-KDRAM) and receivers. Per-call medians, median device."""

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
dur = collections.defaultdict(lambda: collections.defaultdict(list))
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
        zn = p[10]
        if not (zn.startswith("R-") or zn.startswith("IN-")):
            continue
        k = (int(p[1]), int(p[2]), p[3], zn)
        if p[11] == "ZONE_START":
            op[k] = t
        elif p[11] == "ZONE_END" and k in op:
            dur[k[:2]][(p[3], zn)].append((t - op.pop(k)) / 1350)
inj = [c for c in dur if ("NCRISC", "R-KDRAM") in dur[c]]
rcv = [c for c in dur if c not in inj and ("NCRISC", "R-KRECV") in dur[c]]
for label, cs in (("injectors", inj), ("receivers", rcv)):
    print(f"\n{label}: {len(cs)} cores")
    names = sorted({z for c in cs for z in dur[c]})
    for z in names:
        have = [c for c in cs if z in dur[c]]
        per = [st.median(dur[c][z]) for c in have]
        n = [len(dur[c][z]) for c in have]
        print(
            f"  {z[0]:8s} {z[1]:12s} per-call med {st.median(per):7.2f} us   calls/core {st.median(n):5.0f}   cores {len(have)}"
        )
