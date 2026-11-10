"""Per-core duration map of one zone (us) on the logical grid, from profile_log_device.csv. usage: zone_map.py csv ZONE [MHZ]"""
import csv, sys, collections
path, zone = sys.argv[1], sys.argv[2]; MHZ = float(sys.argv[3]) if len(sys.argv) > 3 else 1350.0
with open(path) as f:
    f.readline(); rows = list(csv.reader(f))
cols = [c.strip() for c in rows[0]]; ix = {c: i for i, c in enumerate(cols)}
last = rows[-1][ix["run host ID"]].strip()
ev = collections.defaultdict(lambda: [None, None])
for r in rows[1:]:
    if len(r) < len(cols) - 1 or r[ix["run host ID"]].strip() != last or r[ix["zone name"]].strip() != zone: continue
    k = (int(r[ix["core_x"]]), int(r[ix["core_y"]])); t = int(r[ix["time[cycles since reset]"]]); typ = r[ix["type"]].strip()
    e = ev[k]
    if typ == "ZONE_START": e[0] = t if e[0] is None else min(e[0], t)
    else: e[1] = t if e[1] is None else max(e[1], t)
xs = sorted({k[0] for k in ev}); ys = sorted({k[1] for k in ev})
print(f"zone {zone} run {last} (phys x across, phys y down), us")
print("      " + " ".join(f"{x:>5}" for x in xs))
for y in ys:
    print(f"{y:>5} " + " ".join(f"{(ev[(x,y)][1]-ev[(x,y)][0])/MHZ:5.0f}" if (x, y) in ev and None not in ev[(x, y)] else "    -" for x in xs))
