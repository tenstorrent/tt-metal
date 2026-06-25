# SPDX-License-Identifier: Apache-2.0
"""Parse the latest tracy ops CSV: per-op flow, shapes, core count, latency
for the LAST metal-trace replay session (the profiled single replay)."""
import csv
import glob
import os
import sys

REPORTS = "generated/profiler/reports"


def shp(r, pfx):
    dims = [r.get(f"{pfx}_{a}_PAD[LOGICAL]", "").strip() for a in ("W", "Z", "Y", "X")]
    dims = [d for d in dims if d]
    if not dims:
        return ""
    return "x".join(dims)


def mem(r, pfx):
    m = r.get(f"{pfx}_MEMORY", "").strip()
    m = m.replace("DEV_0_", "").replace("MemoryConfig", "")
    for tok in ("TensorMemoryLayout::", "BufferType::", "dram", "DRAM", "L1"):
        pass
    # compact
    lay = "L1" if "L1" in m else ("DRAM" if "DRAM" in m else "?")
    if "WIDTH_SHARDED" in m:
        lay += "/Wsh"
    elif "BLOCK_SHARDED" in m:
        lay += "/Bsh"
    elif "HEIGHT_SHARDED" in m:
        lay += "/Hsh"
    elif "INTERLEAVED" in m:
        lay += "/int"
    return lay


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        cands = sorted(glob.glob(f"{REPORTS}/*/ops_perf_results_*.csv"), key=os.path.getmtime)
        path = cands[-1]
    print(f"# report: {path}\n")
    rows = list(csv.DictReader(open(path)))

    sid_col = "METAL TRACE REPLAY SESSION ID"
    sessions = sorted({(r.get(sid_col, "") or "").strip() for r in rows} - {""})
    print(f"# replay sessions present: {sessions}")
    target = sessions[-1] if sessions else ""
    sess = [r for r in rows if (r.get(sid_col, "") or "").strip() == target]
    if not sess:
        # no replay session tagging — fall back to device ops only
        sess = [r for r in rows if (r.get("DEVICE KERNEL DURATION [ns]", "") or "").strip()]
    print(f"# profiling replay session = '{target}', {len(sess)} ops\n")

    hdr = f"{'#':>2}  {'OP CODE':<34} {'IN0':<13} {'OUT0':<13} {'cores':>5} {'kern_ns':>9}  in0->out0 mem"
    print(hdr)
    print("-" * len(hdr))
    total = 0
    for i, r in enumerate(sess):
        ns = int(r.get("DEVICE KERNEL DURATION [ns]", "0") or 0)
        total += ns
        cc = r.get("CORE COUNT", "").strip()
        print(
            f"{i:>2}  {r['OP CODE'].strip():<34} {shp(r,'INPUT_0'):<13} "
            f"{shp(r,'OUTPUT_0'):<13} {cc:>5} {ns:>9}  {mem(r,'INPUT_0')}->{mem(r,'OUTPUT_0')}"
        )
    print("-" * len(hdr))
    print(f"{'':>2}  {'TOTAL device-kernel':<34} {'':<13} {'':<13} {'':>5} {total:>9}  ({total/1000:.2f} us)")

    # bucket by op code
    import collections

    bk, ct = collections.Counter(), collections.Counter()
    for r in sess:
        ns = int(r.get("DEVICE KERNEL DURATION [ns]", "0") or 0)
        bk[r["OP CODE"].strip()] += ns
        ct[r["OP CODE"].strip()] += 1
    print("\n# by op code (ns, count):")
    for op, ns in bk.most_common():
        print(f"  {op:<36} {ns:>9} ns  x{ct[op]}  ({100*ns/total:.1f}%)")


if __name__ == "__main__":
    main()
