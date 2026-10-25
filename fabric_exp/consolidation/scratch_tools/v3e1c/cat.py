#!/usr/bin/env python3
"""cat.py VARIANT DIS ELF SCEN: per-category instruction counts of the scenario path (walk -v)."""
import sys, subprocess, re
v, dis, elf, sc = sys.argv[1:5]
out = subprocess.run(["python3", "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3e1c/scen.py", v, dis, elf, sc, "-v"], capture_output=True, text=True).stdout
VAR = {"v3a": "v3a", "v3e1b": "v3l2be1b", "v3e1c": "v3e1c", "v3e1d": "v3e1d"}
cats = {}
rows = []
for l in out.splitlines():
    m = re.match(r"^\s*([0-9a-f]+) (\S+)\s+(.*?)\s+(\S+):(\d+)$", l)
    if not m: continue
    op, f, ln = m.group(2), m.group(4), int(m.group(5))
    if f.startswith("noc_nonblocking_api") and ln in (645, 646, 647, 648, 1173, 1174, 1352, 740): c = "sw-counters"
    elif f.startswith("noc_nonblocking_api") or f.startswith("dataflow_api") or f.startswith("noc_address"): c = "noc-cmd"
    elif f.startswith("fabric_edm_packet_transmission") and 300 <= ln <= 333: c = "impl-dispatch"
    elif f.startswith("fabric_edm_packet_transmission"): c = "arm"
    elif f.startswith("fabric_erisc_router") and ln in (2680,): c = "record-bits"
    else: c = "loop/step:" + f.split(".")[0][-12:]
    cats[c] = cats.get(c, 0) + 1
    rows.append((c, l))
print(v, sc, sum(cats.values()), dict(sorted(cats.items())))
if "-r" in sys.argv:
    for c, l in rows: print(f"  {c:28s} {l}")
