import collections, csv, statistics, sys, json

path, calls = sys.argv[1], int(sys.argv[2])
MHZ = 1350.0
Z = ["HW_SEND", "HR_ARRIVE", "SDPA_HALO_WAIT", "HR_READ"]
starts = {}
data = collections.defaultdict(lambda: collections.defaultdict(list))
kern = collections.defaultdict(lambda: [float("inf"), 0])
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for x in r:
        dev, run, z, k, t = x[0].strip(), x[7].strip(), x[10].strip(), x[11].strip(), int(x[5])
        key = (dev, run)
        if z.endswith("-KERNEL"):
            if k == "ZONE_START":
                kern[key][0] = min(kern[key][0], t)
            elif k == "ZONE_END":
                kern[key][1] = max(kern[key][1], t)
        elif z in Z:
            sk = (key, z, x[1], x[2], x[3])
            if k == "ZONE_START":
                starts[sk] = t
            elif sk in starts:
                data[key][z].append((starts.pop(sk), t))
runs = collections.defaultdict(list)
for d, rn in kern:
    runs[d].append(int(rn))
out = {}
for d, rs in runs.items():
    keep = sorted(rs)[-calls:]
    vals = collections.defaultdict(list)
    for rn in keep:
        key = (d, str(rn))
        t0, t1 = kern[key]
        vals["end"].append((t1 - t0) / MHZ)
        for z in Z:
            sp = data[key].get(z)
            if sp:
                vals[z + "_end"].append(max(e - t0 for _, e in sp) / MHZ)
                vals[z + "_dur"].append(statistics.median(e - s for s, e in sp) / MHZ)
    out[d] = {k: round(statistics.median(v), 1) for k, v in vals.items()}
print(json.dumps(out))
