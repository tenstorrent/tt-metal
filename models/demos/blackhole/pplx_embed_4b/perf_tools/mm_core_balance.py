# Per-core compute-kernel time of a traced minimal_matmul (bench_mm_ablate.py --child under the device profiler):
# TRISC1 (math) kernel duration per core for the last traced call, min / mean / max over cores, and how much of the
# op's span the slowest core accounts for. Tells work imbalance (cores idle while the busiest finishes) apart from
# per-tile kernel cost (mean core cycles / mean tile products per core, M_t * N_t * K_t / cores).
# Usage: mm_core_balance.py <profiler dir> [M_tiles N_tiles K_tiles]
import collections
import csv
import statistics
import sys

d = sys.argv[1]
mnk = [int(v) for v in sys.argv[2:5]] if len(sys.argv) > 4 else None
f = open(f"{d}/.logs/profile_log_device.csv")
next(f)
r = csv.reader(f)
ix = {h.strip(): i for i, h in enumerate(next(r))}
spans = collections.defaultdict(lambda: [1 << 62, 0])
for row in r:
    if not row[ix["zone name"]].strip().endswith("-KERNEL") or row[ix["RISC processor type"]].strip() != "TRISC_1":
        continue
    tid = row[ix["trace id"]].strip()
    key = (
        tid,
        row[ix["trace id counter"]].strip(),
        row[ix["run host ID"]].strip(),
        int(row[ix["core_x"]]),
        int(row[ix["core_y"]]),
    )
    t = int(row[ix["time[cycles since reset]"]])
    s = spans[key]
    s[0], s[1] = min(s[0], t), max(s[1], t)
by_call = collections.defaultdict(dict)
for (tid, cnt, op, x, y), (a, b) in spans.items():
    by_call[(tid, cnt, op)][(x, y)] = b - a
last = sorted(by_call, key=lambda k: (k[0], int(k[1] or 0), int(k[2] or 0)))[-1]
cores = by_call[last]
v = sorted(cores.values())
print(
    f"RES call {last}: {len(v)} cores, math-kernel cycles min {v[0]} mean {statistics.mean(v):.0f} max {v[-1]}"
    f" (mean / max {statistics.mean(v) / v[-1] * 100:.1f}%)"
)
if mnk:
    tp = mnk[0] * mnk[1] * mnk[2] / len(v)
    print(
        f"RES mean core: {statistics.mean(v) / tp:.2f} cycles per tile product over {tp:.0f} tile products"
        f" (LLK 1x8 kt 8: 17.4, kt 40: 17.2); op span at the busiest core {v[-1] / 1350:.1f} us"
    )
