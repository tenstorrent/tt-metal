import csv, collections, glob, os

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


def idx(name):
    return [i for i, r in enumerate(rows) if (r[K_OP] or "").strip() == name]


ds = idx("DENOISE_START")
de = idx("DENOISE_END")
print("DENOISE_START rows:", ds, " DENOISE_END rows:", de)
lo = min(ds) if ds else 0
hi = max(de) if de else len(rows)
region = [r for r in rows[lo:hi] if (r.get(K_ID, "") or "").strip() in ("0", "")]


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
tot_kn = sum(a[2] for a in agg.values())
print(f"=== DENOISE region (device0) ops={sum(a[0] for a in agg.values())} ===")
print(
    f"device_fw={tot_fw/1e6:.1f}ms kernel={tot_kn/1e6:.1f}ms gap={tot_gap/1e6:.1f}ms  GAP%={tot_gap/(tot_fw+tot_gap)*100:.1f}"
)
print(f"{'FW ms':>7} {'%':>5} {'KERN ms':>7} {'GAP ms':>7} {'n':>5} {'avgFW us':>9}  OP")
for op, a in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
    print(
        f"{a[1]/1e6:7.1f} {a[1]/tot_fw*100:5.1f} {a[2]/1e6:7.1f} {a[3]/1e6:7.1f} {a[0]:5d} {a[1]/a[0]/1e3:9.1f}  {op}"
    )
