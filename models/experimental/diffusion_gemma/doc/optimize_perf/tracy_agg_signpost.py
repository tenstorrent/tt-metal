import csv, collections, glob, os, sys

lo_name, hi_name = sys.argv[1], sys.argv[2]
R = sorted(
    glob.glob("/home/zni/tt-metal-tracy/generated/profiler/**/ops_perf_results*.csv", recursive=True),
    key=os.path.getmtime,
)[-1]
rows = list(csv.DictReader(open(R)))
h = rows[0]


def find(*ts):
    for c in h:
        if all(t in c.lower() for t in ts):
            return c


K_OP = find("op", "code")
K_ID = find("device", "id")
K_FW = find("device", "fw", "dur")
K_KN = find("device", "kernel", "dur")
K_GAP = "OP TO OP LATENCY [ns]"


def idxs(name):
    return [i for i, r in enumerate(rows) if (r[K_OP] or "").strip() == name]


lo = idxs(lo_name)
hi = idxs(hi_name)
if not lo or not hi:
    print(f"signpost {lo_name}/{hi_name} not both found: {len(lo)}/{len(hi)}")
    sys.exit()
region = [r for r in rows[min(lo) : max(hi)] if (r.get(K_ID, "") or "").strip() in ("0", "")]


def f(r, k):
    try:
        return float(r[k])
    except:
        return 0.0


agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for r in region:
    op = (r[K_OP] or "").strip() or "(blank)"
    if op.endswith("_START") or op.endswith("_END"):
        continue
    a = agg[op]
    a[0] += 1
    a[1] += f(r, K_FW)
    a[2] += f(r, K_KN)
    a[3] += f(r, K_GAP)
tot_fw = sum(a[1] for a in agg.values())
tot_gap = sum(a[3] for a in agg.values())
print(
    f"=== {lo_name}..{hi_name} (device0) ops={sum(a[0] for a in agg.values())}  fw={tot_fw/1e6:.1f}ms gap={tot_gap/1e6:.1f}ms GAP%={tot_gap/(tot_fw+tot_gap+1e-9)*100:.1f} ==="
)
for op, a in sorted(agg.items(), key=lambda kv: -kv[1][1])[:8]:
    print(f"{a[1]/1e6:7.1f}ms {a[1]/(tot_fw+1e-9)*100:5.1f}%  n={a[0]:5d}  {op}")
