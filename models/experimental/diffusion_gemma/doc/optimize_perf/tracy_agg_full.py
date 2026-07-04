import collections
import csv
import glob
import os
import sys

# Usage: python tracy_agg_full.py LO_SIGNPOST HI_SIGNPOST [divisor] [csv_glob_root]
lo_name, hi_name = sys.argv[1], sys.argv[2]
divisor = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
root = sys.argv[4] if len(sys.argv) > 4 else "/home/zni/tt-metal-tracy/generated/profiler"

R = sorted(glob.glob(f"{root}/**/ops_perf_results*.csv", recursive=True), key=os.path.getmtime)[-1]
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
K_CORE = find("core", "count")


def idxs(name):
    return [i for i, r in enumerate(rows) if (r[K_OP] or "").strip() == name]


lo = idxs(lo_name)
hi = idxs(hi_name)
print(f"csv={os.path.basename(R)}")
print(f"signpost {lo_name} rows={lo[:4]} {hi_name} rows={hi[:4]}")
if not lo or not hi:
    print("SIGNPOST NOT FOUND — aggregating WHOLE run")
    region = [r for r in rows if (r.get(K_ID, "") or "").strip() in ("0", "")]
else:
    region = [r for r in rows[min(lo) : max(hi)] if (r.get(K_ID, "") or "").strip() in ("0", "")]


def f(r, k):
    try:
        return float(r[k])
    except Exception:
        return 0.0


agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
for r in region:
    op = (r[K_OP] or "").strip() or "(blank)"
    if op.endswith("_START") or op.endswith("_END"):
        continue
    a = agg[op]
    a[0] += 1
    a[1] += f(r, K_FW)
    a[2] += f(r, K_KN)
    a[3] += f(r, K_GAP)
    a[4] = max(a[4], f(r, K_CORE) if K_CORE else 0)

tot_fw = sum(a[1] for a in agg.values())
tot_kn = sum(a[2] for a in agg.values())
tot_gap = sum(a[3] for a in agg.values())
print(
    f"=== {lo_name}..{hi_name} region ops={sum(a[0] for a in agg.values())} "
    f"fw={tot_fw/1e6:.1f}ms kernel={tot_kn/1e6:.1f}ms gap={tot_gap/1e6:.1f}ms "
    f"GAP%={tot_gap/(tot_fw+tot_gap+1e-9)*100:.1f}  divisor={divisor} ==="
)
print(f"  per-divisor: fw={tot_fw/1e6/divisor:.2f}ms kernel={tot_kn/1e6/divisor:.2f}ms")
print(f"{'FW ms':>8} {'%fw':>5} {'KERN ms':>8} {'GAP ms':>8} {'n':>6} {'avgFWus':>8} {'cores':>5}  OP")
for op, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    if a[1] / 1e6 < 0.01 and a[0] < 50:
        continue
    print(
        f"{a[1]/1e6:8.2f} {a[1]/(tot_fw+1e-9)*100:5.1f} {a[2]/1e6:8.2f} {a[3]/1e6:8.2f} "
        f"{a[0]:6d} {a[1]/1e3/max(a[0],1):8.1f} {int(a[4]):5d}  {op}"
    )
