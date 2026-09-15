"""Compare two `--profile` CSVs of test_groupnorm_sc_N_1_HW_C_perf_guard.py (rows in execution order).

usage: python3 compare_perf_guard_csv.py <baseline.csv> <new.csv>
"""
import csv
import sys

NAMES = [
    f"{lay}-{aff}-{reg}"
    for lay in ("tile", "rm")
    for aff in ("no_affine", "gamma_beta")
    for reg in ("resident", "streaming", "single_core")
] + [
    "floor_32x32",
    "small_64x64",
    "small_128x128",
    "small_64x320",
    "sd_256x1280",
    "sd_4096x320",
    "sdxl_4096x640",
    "sd_1024x1920",
    "sdxl_16384x320",
]


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    return [(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000.0, int(r["CORE COUNT"])) for r in rows]


a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"{'case':32s} {'base us':>9s} {'cores':>5s} | {'new us':>9s} {'cores':>5s} | {'new/base':>8s}")
for n, (ua, ca), (ub, cb) in zip(NAMES, a, b):
    print(f"{n:32s} {ua:9.2f} {ca:5d} | {ub:9.2f} {cb:5d} | {ub/ua:8.3f}")
