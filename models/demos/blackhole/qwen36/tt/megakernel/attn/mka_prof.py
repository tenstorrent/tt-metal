"""Per-zone device times from profile_log_device.csv (ttsim cycles at 1350 MHz): for each zone name, the program-wide
span (first start .. last end over all cores) and the median / max per-core duration. Also the whole-kernel span."""
import csv, sys, collections, statistics
MHZ = 1350.0
path = sys.argv[1]
with open(path) as f:
    hdr = f.readline(); rows = list(csv.reader(f))
if len(sys.argv) > 2: MHZ = float(sys.argv[2])
else:
    import re
    m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", hdr)
    if m and int(m.group(1)) > 0: MHZ = float(m.group(1))
print(f"freq {MHZ} MHz")
cols = [c.strip() for c in rows[0]]; ix = {c: i for i, c in enumerate(cols)}
runs = collections.OrderedDict()
for r in rows[1:]:
    if len(r) < len(cols) - 1: continue
    rid = r[ix["run host ID"]].strip(); z = r[ix["zone name"]].strip(); t = int(r[ix["time[cycles since reset]"]]); typ = r[ix["type"]].strip()
    key = (r[ix["core_x"]], r[ix["core_y"]], r[ix["RISC processor type"]].strip(), z)
    d = runs.setdefault(rid, collections.defaultdict(list))
    d[key].append((typ, t))
for rid, d in runs.items():
    span = collections.defaultdict(lambda: [None, None]); per = collections.defaultdict(list)
    for (x, y, risc, z), ev in d.items():
        st = [t for typ, t in ev if typ == "ZONE_START"]; en = [t for typ, t in ev if typ == "ZONE_END"]
        if not st or not en: continue
        s, e = min(st), max(en); sp = span[z]
        sp[0] = s if sp[0] is None else min(sp[0], s); sp[1] = e if sp[1] is None else max(sp[1], e)
        per[z].append(e - s)
    print(f"run {rid}")
    for z, (s, e) in sorted(span.items(), key=lambda kv: kv[1][0]):
        p = per[z]
        print(f"  {z:<22} span {(e-s)/MHZ:9.2f} us  per-core median {statistics.median(p)/MHZ:8.2f} max {max(p)/MHZ:8.2f}  n={len(p)}")
