# Device kernel time per op call of a traced bench run under the device profiler, one line per trace (in capture
# order), so wall-clock bench variants can be read in the same unit as a tracy profile's DEVICE KERNEL DURATION:
# first kernel start to last kernel end over all cores, in cycles at the nominal 1.35 GHz (no op-to-op dispatch gap).
# Usage: TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir> python <bench> ...
#        device_kernel_us.py <dir> [label ...]   (labels name the traces in order, e.g. the bench's variant names)
import collections
import csv
import statistics
import sys


def kernel_us(d):
    """{trace id: [µs per call]} from <d>/.logs/profile_log_device.csv"""
    f = open(f"{d}/.logs/profile_log_device.csv")
    next(f)
    r = csv.reader(f)
    ix = {h.strip(): i for i, h in enumerate(next(r))}
    lo, hi = collections.defaultdict(lambda: 1 << 62), collections.defaultdict(int)
    for row in r:
        if not row[ix["zone name"]].strip().endswith("-KERNEL"):
            continue
        tid = row[ix["trace id"]].strip()
        if not tid:
            continue
        key = (int(tid), row[ix["trace id counter"]].strip(), row[ix["run host ID"]].strip())
        t = int(row[ix["time[cycles since reset]"]])
        lo[key], hi[key] = min(lo[key], t), max(hi[key], t)
    by = collections.defaultdict(list)
    for key in lo:
        by[key[0]].append((hi[key] - lo[key]) / 1350)
    return dict(sorted(by.items()))


if __name__ == "__main__":
    labels = sys.argv[2:]
    for i, (tid, us) in enumerate(kernel_us(sys.argv[1]).items()):
        name = labels[i] if i < len(labels) else f"trace {tid}"
        print(f"RES {name:28s} {statistics.median(us):7.1f} us/call device (n={len(us)})")
