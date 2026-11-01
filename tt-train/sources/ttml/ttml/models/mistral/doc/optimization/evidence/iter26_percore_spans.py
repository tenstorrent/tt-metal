import csv, sys, collections, statistics as st

f = sys.argv[1]
ops = list(csv.DictReader(open(sys.argv[2])))
runid = {}
for r in ops:
    try:
        runid[int(r["GLOBAL CALL COUNT"])] = r["OP CODE"]
    except:
        pass
lines = open(f).read().splitlines()
hdr_i = [i for i, l in enumerate(lines) if l.startswith("PCIe slot")][0]
z = collections.defaultdict(lambda: [None, None])
for row in csv.reader(lines[hdr_i + 1 :]):
    if len(row) < 12:
        continue
    core = (int(row[1]), int(row[2]))
    risc = row[3].strip()
    t = int(row[5])
    run = int(row[7])
    zn = row[10].strip()
    ph = row[11].strip()
    if not zn.endswith("KERNEL"):
        continue
    k = (run, core, risc)
    if ph == "ZONE_START":
        z[k][0] = t
    else:
        z[k][1] = t
per_run = collections.defaultdict(dict)
for (run, core, risc), (s, e) in z.items():
    if s is None or e is None:
        continue
    per_run[run].setdefault(core, {})[risc] = (s, e)
by_op = collections.defaultdict(list)
for run, cores in per_run.items():
    name = runid.get(run, "?")
    if "SDPA" not in name:
        continue
    spans = []
    starts = []
    ends = []
    for core, rs in cores.items():
        s = min(v[0] for v in rs.values())
        e = max(v[1] for v in rs.values())
        spans.append((e - s) / 1350)
        starts.append(s)
        ends.append(e)
    spans.sort()
    by_op[name].append(
        (
            len(cores),
            spans[0],
            st.median(spans),
            spans[-1],
            (max(ends) - min(starts)) / 1350,
            (max(starts) - min(starts)) / 1350,
        )
    )
for name, v in by_op.items():
    v = v[3:]
    print(name, "runs", len(v), "cores", v[0][0])
    print(
        "   per-core span us: min %.1f median %.1f max %.1f | op span %.1f | start skew %.1f"
        % tuple(st.median(x[i] for x in v) for i in range(1, 6))
    )
sel = [run for run in per_run if sys.argv[3] in runid.get(run, "")]
run = sel[-1]
cores = per_run[run]
t0 = min(min(v[0] for v in rs.values()) for rs in cores.values())
rows = []
for core, rs in sorted(cores.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    s = min(v[0] for v in rs.values())
    e = max(v[1] for v in rs.values())
    rows.append(
        (
            core,
            round((s - t0) / 1350, 1),
            round((e - t0) / 1350, 1),
            {r: round((v[1] - v[0]) / 1350, 1) for r, v in rs.items()},
        )
    )
print(sys.argv[3], "run", run)
for r in rows[:16]:
    print("  ", r)
print("end us sorted:", sorted(r[2] for r in rows)[:5], "...", sorted(r[2] for r in rows)[-8:])
