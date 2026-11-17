# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the zone profile (profile_log_device.csv) of test_zone_probe: per call, per chip, per zone: start/end
relative to that chip's earliest kernel start of the call (us), min/median/max over cores."""
import csv, sys, statistics as st
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
f = open(path)
first = f.readline()
freq = 1.35
if "CHIP_FREQ" in first:
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0]) / 1000.0
rd = csv.reader(f)
hdr = [h.strip() for h in next(rd)]
ix = {h: i for i, h in enumerate(hdr)}
rows = list(rd)
dev_k = next(k for k in ix if k.startswith("PCIe slot"))
ev = defaultdict(list)  # (dev, run_id) -> list
for r in rows:
    name = r[ix["zone name"]].strip()
    typ = r[ix["type"]].strip()
    ev[(r[ix[dev_k]].strip(), r[ix["run host ID"]].strip())].append(
        (
            int(r[ix["time[cycles since reset]"]]),
            name,
            typ,
            r[ix["core_x"]],
            r[ix["core_y"]],
            r[ix["RISC processor type"]].strip(),
        )
    )
# keep only runs that contain our zones, ordered per device
runs = defaultdict(list)
for (dev, rid), es in ev.items():
    if any(n.startswith(("ps_", "pr_", "rr_", "rc_", "rw_")) for _, n, *_ in es):
        runs[dev].append((min(t for t, *_ in es), rid, es))
for dev in sorted(runs):
    rs = sorted(runs[dev])
    last = rs[-1]  # last call of the trace
    t0 = min(t for t, n, typ, *_ in last[2] if "KERNEL" in n.upper() or typ == "ZONE_START")
    kstart = [t for t, n, typ, *_ in last[2] if n.endswith("-KERNEL") and typ == "ZONE_START"]
    kend = [t for t, n, typ, *_ in last[2] if n.endswith("-KERNEL") and typ == "ZONE_END"]
    t0 = min(kstart) if kstart else t0
    print(
        f"=== device {dev}: {len(rs)} calls, last call span {(max(kend) - t0) / freq / 1000:.2f} us"
        if kend
        else f"=== device {dev}"
    )
    open_ = {}
    spans = defaultdict(list)
    for t, n, typ, x, y, risc in sorted(last[2]):
        k = (n, x, y, risc)
        if typ == "ZONE_START":
            open_[k] = t
        elif typ == "ZONE_END" and k in open_:
            spans[n].append(((open_.pop(k) - t0) / freq / 1000, (t - t0) / freq / 1000))
    for n in sorted(spans, key=lambda n: st.median(s for s, _ in spans[n])):
        ss = [s for s, _ in spans[n]]
        es = [e for _, e in spans[n]]
        du = [e - s for s, e in spans[n]]
        print(
            f"  {n:28s} n={len(ss):4d} start {min(ss):6.2f}/{st.median(ss):6.2f}/{max(ss):6.2f}  end {min(es):6.2f}/{st.median(es):6.2f}/{max(es):6.2f}  dur med {st.median(du):5.2f} max {max(du):5.2f}"
        )
