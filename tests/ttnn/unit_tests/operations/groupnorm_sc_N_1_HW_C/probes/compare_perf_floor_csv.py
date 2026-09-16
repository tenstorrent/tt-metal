"""Compare two `--profile` CSVs of test_groupnorm_sc_N_1_HW_C_perf_floor.py (rows in execution order).

usage: python3 compare_perf_floor_csv.py <baseline.csv> <new.csv>
"""
import csv
import sys

NAMES = [
    "floor_32x32_g1",
    "small_64x64_g2",
    "small_128x128_g4",
    "small_64x320_g32",
    "multi_8x64x64_g2",
    "single_core_8x64x160_g8_cap4",
    "sentinel_1024x640",
    "sentinel_16384x320",
    "floor_32x32_g1_no_affine",
    "small_64x320_g32_no_affine",
]


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    return [(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000.0, int(r["CORE COUNT"])) for r in rows]


a = load(sys.argv[1])
b = load(sys.argv[2]) if len(sys.argv) > 2 else None
print(f"{'case':32s} {'base us':>9s} {'cores':>5s} | {'new us':>9s} {'cores':>5s} | {'new/base':>8s}")
for i, n in enumerate(NAMES):
    if i >= len(a):
        break
    ua, ca = a[i]
    if b is None or i >= len(b):
        print(f"{n:32s} {ua:9.2f} {ca:5d} |")
        continue
    ub, cb = b[i]
    print(f"{n:32s} {ua:9.2f} {ca:5d} | {ub:9.2f} {cb:5d} | {ub/ua:8.3f}")
