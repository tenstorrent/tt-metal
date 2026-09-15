import collections
import statistics

MHZ = 1350.0


def summarize(log, runid, label):
    ev = collections.defaultdict(dict)
    with open(log) as f:
        next(f)
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 12 or p[7].strip() != runid:
                continue
            if not p[10].strip().endswith("-KERNEL"):
                continue
            ev[(int(p[0]), int(p[1]), int(p[2]), p[3].strip())][p[11].strip()] = int(p[5])
    out = []
    for dev in sorted({k[0] for k in ev}):
        cores = collections.defaultdict(dict)
        for (d, x, y, r), z in ev.items():
            if d == dev and "ZONE_START" in z and "ZONE_END" in z:
                cores[(x, y)][r] = (z["ZONE_START"], z["ZONE_END"])
        t0 = min(s for c in cores.values() for (s, e) in c.values())
        end = lambda c: (max(e for (s, e) in cores[c].values()) - t0) / MHZ / 1e3
        vsa = [c for c in cores if any(r.startswith("TRISC") for r in cores[c])]
        snd = [c for c in cores if c not in vsa]
        ve = sorted(end(c) for c in vsa)
        se = sorted(end(c) for c in snd)
        out.append((dev, len(snd), se[-1] if se else float("nan"), ve[0], statistics.median(ve), ve[-1]))
    dev, ns, s_end, v0, vm, v1 = max(out, key=lambda t: t[5])
    print(
        f"{label:<44} dev {dev:2d} senders={ns:2d} gather end {s_end:6.2f} | VSA end min/med/max {v0:6.2f} {vm:6.2f} {v1:6.2f}"
    )


runs = [
    ("2026_09_15_18_44_31", "93200", "stock gather, per-shard gate (default)"),
    ("2026_09_14_23_43_00", "93200", "stock gather, gate open"),
    ("2026_09_14_23_39_01", "93200", "stock gather, serialized (WAIT_ALL)"),
    ("2026_09_15_00_24_48", "93200", "fused gather (MUX x2), per-shard gate"),
    ("2026_09_11_18_53_14", "95248", "fused gather, per-block gate (18/10)"),
    ("2026_09_11_18_31_33", "95248", "fused gather, gate open"),
    ("2026_09_11_18_49_10", "95248", "fused gather, serialized (WAIT_ALL)"),
]
for d, rid, label in runs:
    summarize(f"generated/profiler/reports/{d}/profile_log_device.csv", rid, label)
