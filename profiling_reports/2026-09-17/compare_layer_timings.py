"""Per-device Dispatch/Combine/FFN durations for chosen layers, from an ops_perf_results CSV.

Rows are in host-enqueue order, so the signpost rows delimit layers: everything between
forward_layer_<N>_start and _end belongs to layer N. Each layer runs twice (a WARMUP pass from
runtime.compile() and the measured chunk); pass 1 carries kernel-compile cost, so only pass 2 is
reported.
"""
import collections
import csv
import sys

OPS = {
    "DispatchDeviceOperation": "Dispatch",
    "CombineDeviceOperation": "Combine",
    "UnifiedRoutedExpertFfnDeviceOperation": "FFN",
}


def read(path, layers):
    rows = collections.defaultdict(dict)  # (layer, op) -> {device: us}
    seen = collections.Counter()
    layer = None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            code = (r.get("OP CODE") or "").strip()
            if code.startswith("forward_layer_"):
                n = int(code.split("_")[2])
                if code.endswith("_start"):
                    layer, seen[n] = n, seen[n] + 1
                else:
                    layer = None
                continue
            if layer not in layers or code not in OPS or seen[layer] != 2:
                continue
            dev = (r.get("DEVICE ID") or "").strip()
            dur = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
            if dev and dur:
                rows[(layer, OPS[code])][int(dev)] = float(dur) / 1e6
    return rows


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    layers = [int(x) for x in sys.argv[3].split(",")]
    a, b = read(a_path, set(layers)), read(b_path, set(layers))

    print(
        f"{'layer/op':14} {'baseline max':>13} {'collector max':>14} {'delta':>9}   "
        f"{'baseline mean':>14} {'collector mean':>15} {'delta':>9}"
    )
    for layer in layers:
        for op in ("Dispatch", "Combine", "FFN"):
            av, bv = a.get((layer, op)), b.get((layer, op))
            if not av or not bv:
                print(f"L{layer} {op:9} MISSING (baseline={bool(av)} collector={bool(bv)})")
                continue
            amax, bmax = max(av.values()), max(bv.values())
            amean, bmean = sum(av.values()) / len(av), sum(bv.values()) / len(bv)
            print(
                f"L{layer} {op:9} {amax:10.3f} ms {bmax:11.3f} ms {100*(bmax-amax)/amax:+8.2f}%   "
                f"{amean:11.3f} ms {bmean:12.3f} ms {100*(bmean-amean)/amean:+8.2f}%"
            )
        print()

    for label, d in (("baseline", a), ("collector", b)):
        for layer in layers:
            for op in ("Dispatch", "Combine", "FFN"):
                v = d.get((layer, op))
                if v and len(v) != 8:
                    print(f"WARNING {label} L{layer} {op}: {len(v)} devices, expected 8")


main()
