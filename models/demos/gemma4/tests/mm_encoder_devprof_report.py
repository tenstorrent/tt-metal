# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-op DEVICE time of the multimodal encoder unit profiled by
``mm_encoder_perf_probe.py`` (PROBE_PROFILE=1 under ``python -m tracy -p --no-web-server``).

The device rows (``cpp_device_perf_report.csv``) carry no op names on this build.
The probe graph-captured one replay of the unit; its depth-1 device operations
(every entry except ``Tensor::deallocate`` / ``Tensor::reshape``, which launch no
program) are exactly the rows one replay produces, so the last ``calls x ops``
rows of a device are named by order. Kernel durations are the honest device time;
FW durations include dispatch waits (an op's firmware can start before the
previous op's cores are done).

    python mm_encoder_devprof_report.py <cpp_device_perf_report.csv> <probe_ops.json> --calls 5 [--units 27]
"""
import argparse
import csv
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("ops_json")
    ap.add_argument("--calls", type=int, required=True, help="profiled replays of the unit (PROBE_ITERS)")
    ap.add_argument("--units", type=int, default=1, help="units per encoder call (blocks per tower)")
    ap.add_argument("--devices", default="0,1,2,3")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    cols = list(rows[0].keys())
    col = lambda n: next(c for c in cols if c.strip() == n)
    dev, gc = col("DEVICE ID"), col("GLOBAL CALL COUNT")
    fw_c, kn_c = col("DEVICE FW DURATION [ns]"), col("DEVICE KERNEL DURATION [ns]")
    seq = [
        o["name"]
        for o in json.load(open(args.ops_json))["ops"]
        if o["depth"] == 1 and o["name"] not in ("Tensor::deallocate", "Tensor::reshape")
    ]
    n = len(seq)
    fw, kn, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    ndev = 0
    for devid in args.devices.split(","):
        r = sorted([x for x in rows if x[dev].strip() == devid], key=lambda x: int(x[gc]))
        if len(r) < args.calls * n:
            continue
        ndev += 1
        tail = r[-args.calls * n :]
        for k in range(args.calls):
            for name, x in zip(seq, tail[k * n : (k + 1) * n]):
                fw[name] += float(x[fw_c]) / 1e6
                kn[name] += float(x[kn_c] or 0) / 1e6
                if k == 0 and ndev == 1:
                    cnt[name] += 1
    den = args.calls * ndev
    print(
        f"unit: {n} device ops, {args.calls} replays x {ndev} devices averaged; {args.units} unit(s) per encoder call"
    )
    print(f"{'op':40s} {'x':>3s} {'FW ms':>8s} {'kernel ms':>10s} {'per call':>10s}")
    tot_fw = tot_kn = 0.0
    for name in sorted(kn, key=lambda k: -kn[k]):
        f, k = fw[name] / den, kn[name] / den
        tot_fw += f
        tot_kn += k
        print(f"{name:40s} {cnt[name]:3d} {f:8.3f} {k:10.3f} {args.units * k:10.2f}")
    print(f"{'TOTAL per unit':40s} {sum(cnt.values()):3d} {tot_fw:8.3f} {tot_kn:10.3f} {args.units * tot_kn:10.2f}")


if __name__ == "__main__":
    main()
