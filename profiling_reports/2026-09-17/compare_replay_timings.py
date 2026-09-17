"""Per-iteration max-across-device duration for Dispatch/Combine, first iteration discarded."""
import csv, collections, statistics, sys, glob, os


def read(run_dir):
    csvs = glob.glob(os.path.join(run_dir, "profiler/reports/*/ops_perf_results*.csv"))
    inst = collections.defaultdict(dict)
    order = collections.defaultdict(list)
    for r in csv.DictReader(open(sorted(csvs)[-1])):
        op = (r.get("OP CODE") or "").strip()
        if op not in ("DispatchDeviceOperation", "CombineDeviceOperation"):
            continue
        dev = int((r.get("DEVICE ID") or "0").strip())
        dur = float((r.get("DEVICE KERNEL DURATION [ns]") or 0)) / 1e6
        gcc = int((r.get("GLOBAL CALL COUNT") or 0))
        inst[op][gcc] = (dev, dur)
    out = {}
    for op, d in inst.items():
        # one logical instance = 8 consecutive call counts, one per device
        by_inst = collections.defaultdict(list)
        for gcc, (dev, dur) in d.items():
            by_inst[gcc - dev].append(dur)
        iters = [max(v) for _, v in sorted(by_inst.items())]
        out[op] = iters[1:]  # discard the first iteration
    return out


a, b = read(sys.argv[1]), read(sys.argv[2])
print(f"{'op':10}{'captured':>12}{'permuted':>12}{'delta':>10}{'captured sd':>14}{'permuted sd':>13}")
for op, short in (("DispatchDeviceOperation", "Dispatch"), ("CombineDeviceOperation", "Combine")):
    x, y = a[op], b[op]
    mx, my = statistics.mean(x), statistics.mean(y)
    print(
        f"{short:10}{mx:9.3f} ms{my:9.3f} ms{100*(my-mx)/mx:+9.2f}%"
        f"{statistics.stdev(x):11.3f} ms{statistics.stdev(y):10.3f} ms   (n={len(x)},{len(y)})"
    )
