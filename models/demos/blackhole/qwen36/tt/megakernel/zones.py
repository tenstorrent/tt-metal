"""Per-zone timeline for one program run: for each zone name, per occurrence index (layer), median/min/max start and
end (us from kernel start) over cores. usage: zones.py csv runID [mhz]"""
import csv, sys, collections, statistics as st
path, rid = sys.argv[1], sys.argv[2]; mhz = float(sys.argv[3]) if len(sys.argv) > 3 else 1350.0
import os; RISC = os.environ.get("ZRISC")  # e.g. TRISC_0 / TRISC_2 / BRISC / NCRISC
rows = list(csv.reader(open(path)))[1:]; cols = [c.strip() for c in rows[0]]; ix = {c: i for i, c in enumerate(cols)}
ev = collections.defaultdict(list); t0 = None
for r in rows[1:]:
    if len(r) < len(cols) - 1 or r[ix["run host ID"]].strip() != rid: continue
    z = r[ix["zone name"]].strip(); t = int(r[ix["time[cycles since reset]"]]); typ = r[ix["type"]].strip()
    core = (r[ix["core_x"]].strip(), r[ix["core_y"]].strip(), r[ix["RISC processor type"]].strip())
    if z.endswith("-KERNEL") and typ == "ZONE_START": t0 = t if t0 is None else min(t0, t)
    if RISC and core[2] != RISC and not z.endswith("-KERNEL"): continue
    ev[(z, core)].append((typ, t))
occ = collections.defaultdict(lambda: collections.defaultdict(list))
for (z, core), lst in ev.items():
    if z.endswith("-KERNEL") or z.endswith("-FW"): continue
    starts = [t for ty, t in lst if ty == "ZONE_START"]; ends = [t for ty, t in lst if ty == "ZONE_END"]
    for i, (s, e) in enumerate(zip(starts, ends)): occ[z][i].append(((s - t0) / mhz, (e - t0) / mhz))
for z in sorted(occ):
    for i in sorted(occ[z]):
        v = occ[z][i]; s = [a for a, b in v]; e = [b for a, b in v]; d = [b - a for a, b in v]
        print(f"{z:12s} #{i}: start med {st.median(s):7.1f} end med {st.median(e):7.1f} max {max(e):7.1f}  dur med {st.median(d):6.1f} max {max(d):6.1f}  n={len(v)}")
