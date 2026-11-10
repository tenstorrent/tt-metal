"""Per-core duration (us) of the k-th occurrence of a zone (default k=0), on the phys grid. usage: mka2_coremap.py csv ZONE [k]"""
import csv, sys, collections
path, zone = sys.argv[1], sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else 0
with open(path) as f: f.readline(); rows = list(csv.reader(f))
cols = [c.strip() for c in rows[0]]; ix = {c: i for i, c in enumerate(cols)}
last = rows[-1][ix["run host ID"]].strip(); occ = collections.defaultdict(list); pend = {}; t0 = None
for r in rows[1:]:
    if len(r) < len(cols) - 1 or r[ix["run host ID"]].strip() != last: continue
    z = r[ix["zone name"]].strip(); t = int(r[ix["time[cycles since reset]"]]); typ = r[ix["type"]].strip()
    if z.endswith("-KERNEL") and typ == "ZONE_START": t0 = t if t0 is None else min(t0, t)
    if z != zone: continue
    c = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
    if typ == "ZONE_START": pend[c] = t
    elif c in pend: occ[c].append((pend.pop(c), t))
xs = sorted({c[0] for c in occ}); ys = sorted({c[1] for c in occ})
print(f"{zone} occurrence {k}: duration us / end us (phys x across)")
print("      " + " ".join(f"{x:>9}" for x in xs))
for y in ys:
    cells = []
    for x in xs:
        l = sorted(occ.get((x, y), []))
        cells.append(f"{(l[k][1]-l[k][0])/1350:4.0f}/{(l[k][1]-t0)/1350:4.0f}" if len(l) > k else "        -")
    print(f"{y:>5} " + " ".join(cells))
