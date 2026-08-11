#!/usr/bin/env python3
"""Attribute the AGMM waiting term per core, from TT_AGMM_PROFILE_STEPS=1 counters.

The whole-op median can say "13 us of waiting" but not WHERE, so the kernels emit cycles-blocked counters
per wait site via DeviceTimestampedData:

    agmm_wait_own    writer, NCRISC   blocked for this core's own chunk to arrive over the FABRIC
    agmm_wait_ring   writer, NCRISC   blocked for a chunk to arrive over the ON-CHIP ring
    agmm_wait_in0    compute, TRISC_0 compute STARVED on cb0 (the gathered activation)
    agmm_wait_in1    compute, TRISC_0 compute STARVED on cb1 (in1 streaming from DRAM)

TRISC_0 is the unpacker: it is the TRISC that actually blocks on operand availability, so TRISC_1/2 read ~0
and are ignored.

Reading the numbers: these are PER-CORE totals over one invocation, and all 80 cores run concurrently, so
they do not sum to the op's wall time. What matters is (a) in0 vs in1 -- is compute starved on the gather or
on DRAM -- and (b) the spread across cores, since the makespan is set by the worst core.

usage: agmm_stall_report.py [profile_log_device.csv]
"""
import csv
import statistics
import sys

FREQ = 1.35e9
CSV = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
METRICS = ["agmm_send", "agmm_wait_own", "agmm_wait_ring", "agmm_wait_in0", "agmm_wait_in1"]


def main():
    with open(CSV) as fh:
        rows = list(csv.reader(fh))
    hdr = next((i for i, r in enumerate(rows) if "zone name" in [c.strip().lower() for c in r]), None)
    if hdr is None:
        sys.exit(f"no header in {CSV}")
    idx = {c.strip().lower(): i for i, c in enumerate(rows[hdr])}
    # metric -> (device, core) -> cycles. Keyed per core so the spread is visible; TRISC_1/2 dropped.
    data = {m: {} for m in METRICS}
    for r in rows[hdr + 1 :]:
        if len(r) <= max(idx.values()):
            continue
        name = r[idx["zone name"]].strip()
        if name not in data:
            continue
        risc = r[idx["risc processor type"]].strip()
        if risc.startswith("TRISC") and risc != "TRISC_0":
            continue
        key = (r[idx["pcie slot"]].strip(), r[idx["core_x"]].strip(), r[idx["core_y"]].strip())
        # last invocation wins; the hash worker runs the op once
        data[name][key] = int(r[idx["data"]])

    print(f"{'metric':16s} {'cores':>6} {'min':>9} {'median':>9} {'p90':>9} {'max':>9}   (us blocked per core)")
    for m in METRICS:
        v = sorted(x / FREQ * 1e6 for x in data[m].values())
        if not v:
            print(f"{m:16s}   (no samples)")
            continue
        p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
        print(f"{m:16s} {len(v):>6} {v[0]:>9.1f} {statistics.median(v):>9.1f} {p90:>9.1f} {v[-1]:>9.1f}")

    # BLOCKED TIME IS NOT IDLE TIME. TRISC_0 blocked on cb0 does not mean the core is idle -- math and pack
    # keep draining earlier blocks. So the counters above attribute CAUSE, and the per-core KERNEL durations
    # below attribute COST: the makespan is set by whichever core finishes last.
    dur = {}
    for r in rows[hdr + 1 :]:
        if len(r) <= max(idx.values()):
            continue
        zone = r[idx["zone name"]].strip()
        if not zone.endswith("-KERNEL"):
            continue
        risc = r[idx["risc processor type"]].strip()
        key = (r[idx["pcie slot"]].strip(), r[idx["core_x"]].strip(), r[idx["core_y"]].strip(), risc)
        t = int(r[idx[next(c for c in idx if c.startswith("time[cycles"))]])
        typ = r[idx["type"]].strip()
        # LAST zone pair, not the max: the CSV holds every op in the process, and the setup from_torch writes
        # would otherwise dominate. The AGMM is the final op the hash worker runs.
        e = dur.setdefault(key, {})
        if typ == "ZONE_START":
            e["s"] = t
        elif typ == "ZONE_END" and "s" in e:
            e["d"] = t - e["s"]
    per_risc = {}
    for (dev, cx, cy, risc), e in dur.items():
        if "d" in e:
            per_risc.setdefault(risc, []).append(e["d"] / FREQ * 1e6)
    # COLD-START CAVEAT: the hash worker runs ONE invocation with no warmup, so cores start at very different
    # times and a kernel zone can span until the slowest core finishes. These durations are useful for the
    # SPREAD across cores, not as an estimate of the steady-state op time (which the bench worker measures).
    # BRISC/NCRISC counts exceed the 80 matmul cores because the mux cores run those RISCs too.
    print(f"\n{'kernel duration':16s} {'cores':>6} {'min':>9} {'median':>9} {'max':>9}   (us per core, COLD)")
    for risc in sorted(per_risc):
        v = sorted(per_risc[risc])
        print(f"{risc:16s} {len(v):>6} {v[0]:>9.1f} {statistics.median(v):>9.1f} {v[-1]:>9.1f}")

    # The makespan is set by the worst core, so show which cores starve compute most on the gather.
    worst = sorted(data["agmm_wait_in0"].items(), key=lambda kv: -kv[1])[:6]
    print("\nworst cores by compute-starved-on-in0:")
    for (dev, cx, cy), cyc in worst:
        ring = data["agmm_wait_ring"].get((dev, cx, cy), 0) / FREQ * 1e6
        own = data["agmm_wait_own"].get((dev, cx, cy), 0) / FREQ * 1e6
        in1 = data["agmm_wait_in1"].get((dev, cx, cy), 0) / FREQ * 1e6
        print(
            f"  dev {dev:>2} core ({cx},{cy}): in0 {cyc / FREQ * 1e6:6.1f}  in1 {in1:6.1f}  "
            f"| writer own {own:6.1f}  ring {ring:6.1f}"
        )


main()
