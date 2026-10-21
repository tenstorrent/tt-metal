"""Steady-state per-step breakdown from the GACC-* timestamped data of the SDPA op on the median device (TRISC_0/1/2)."""

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
val = collections.defaultdict(dict)  # (risc, name) -> core -> value
with open(glob.glob(R + "/profile_log_device.csv")[0]) as fh:
    next(fh)
    next(fh)
    for line in fh:
        p = line.split(",", 14)
        if p[0] != d or not p[10].startswith("GACC-"):
            continue
        t = int(p[5])
        if t < a or t > b + 200000:
            continue
        val[(p[3], p[10])][(p[1], p[2])] = int(p[6])
for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
    n_all = val.get((risc, "GACC-N-ALL"), {})
    n_late = val.get((risc, "GACC-N-LATE"), {})
    cores = [c for c in n_all if n_all[c] > 0]
    if not cores:
        continue
    print(
        f"\n{risc}: {len(cores)} cores, steps/core med {st.median(n_all[c] for c in cores)}, late steps med {st.median(n_late[c] for c in cores)}"
    )
    for nm in ("STEP", "WKT", "P2", "WV", "SALAD"):
        for tag, n in (("ALL", n_all), ("LATE", n_late)):
            v = val.get((risc, f"GACC-{nm}-{tag}"), {})
            per = [v[c] / max(1, n[c]) / 1350 for c in cores if c in v and n[c] > 0]
            tot = [v[c] / 1350 for c in cores if c in v]
            if per:
                print(
                    f"  {nm:6s} {tag:4s} per-step med {st.median(per):6.2f} us  (per-core total med {st.median(tot):8.1f} us)"
                )
