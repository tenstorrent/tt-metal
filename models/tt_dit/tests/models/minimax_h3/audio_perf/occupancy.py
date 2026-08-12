"""Full time accounting for one decode's op stream: is the device busy or waiting?

analyze_csv.py sums DEVICE FW DURATION and stops there, which is how "1401 ms of device time
against 1.1 s of wall" got quoted -- a sum with no denominator. The CSV also carries HOST START/END
TS and OP TO OP LATENCY, so the same rows can answer the question the sum cannot:

    span      = last HOST END TS - first HOST START TS   (wall clock over the op stream)
    busy      = sum(DEVICE FW DURATION)                  (device firmware actually running)
    gaps      = sum(OP TO OP LATENCY)                    (device idle between ops)

busy/span near 1.0 means the device is saturated and op cost is the wall -- kernel-bound.
busy/span near 0.2 means the device idles between ops waiting on host dispatch -- host-bound.
busy + gaps ~= span is the check that the accounting is complete rather than double-counting.
"""

import csv
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))


def f(r, k):
    try:
        return float(r.get(k) or 0)
    except ValueError:
        return 0.0


# Signposts appear as their own rows in the op stream; keep only what lies between them.
start = end = None
for i, r in enumerate(rows):
    code = (r.get("OP CODE") or "").strip()
    if code == "start" and start is None:
        start = i
    elif code == "stop":
        end = i
win = rows[start + 1 : end] if start is not None and end is not None else rows
print(f"rows total {len(rows)}, window {len(win)} (start={start}, stop={end})")

# Device timestamps are per-device cycle counters; host TS is a single monotonic ns clock, so the
# span has to come from host TS to be comparable across ops.
hs = [f(r, "HOST START TS") for r in win if f(r, "HOST START TS")]
he = [f(r, "HOST END TS") for r in win if f(r, "HOST END TS")]
span_ms = (max(he) - min(hs)) / 1e6 if hs and he else float("nan")

fw_ms = sum(f(r, "DEVICE FW DURATION [ns]") for r in win) / 1e6
kern_ms = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in win) / 1e6
gap_ms = sum(f(r, "OP TO OP LATENCY [ns]") for r in win) / 1e6
host_ms = sum(f(r, "HOST DURATION [ns]") for r in win) / 1e6
disp_ms = sum(f(r, "DISPATCH TOTAL CQ CMD OP TIME [ns]") for r in win) / 1e6
gosend_ms = sum(f(r, "DISPATCH GO SEND WAIT TIME [ns]") for r in win) / 1e6

print(f"\nops in window                {len(win):>10}")
print(f"host-clock span              {span_ms:>10.1f} ms   <- wall time of the op stream")
print(f"device FW busy               {fw_ms:>10.1f} ms   {100 * fw_ms / span_ms:>5.1f}% of span")
print(f"device kernel busy           {kern_ms:>10.1f} ms   {100 * kern_ms / span_ms:>5.1f}% of span")
print(f"op-to-op gaps (device idle)  {gap_ms:>10.1f} ms   {100 * gap_ms / span_ms:>5.1f}% of span")
print(f"  FW + gaps                  {fw_ms + gap_ms:>10.1f} ms   {100 * (fw_ms + gap_ms) / span_ms:>5.1f}% of span")
print(f"host time inside op calls    {host_ms:>10.1f} ms   {100 * host_ms / span_ms:>5.1f}% of span")
print(f"dispatch CQ cmd time         {disp_ms:>10.1f} ms")
print(f"dispatch go-send wait        {gosend_ms:>10.1f} ms")


# Per-op medians say whether the "180 us/op floor" is FW cost or gap cost.
def med(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2] if vals else 0.0


fw_us = [f(r, "DEVICE FW DURATION [ns]") / 1e3 for r in win]
gap_us = [f(r, "OP TO OP LATENCY [ns]") / 1e3 for r in win]
host_us = [f(r, "HOST DURATION [ns]") / 1e3 for r in win]
print(f"\nper-op median: FW {med(fw_us):.1f} us | gap {med(gap_us):.1f} us | host {med(host_us):.1f} us")
print(f"per-op mean:   FW {fw_ms * 1e3 / len(win):.1f} us | gap {gap_ms * 1e3 / len(win):.1f} us")

# Which ops carry the idle, not just the busy time -- a cheap op with a big gap is a dispatch problem,
# not a kernel problem.
agg = defaultdict(lambda: [0, 0.0, 0.0])
for r in win:
    a = agg[(r.get("OP CODE") or "?").strip()]
    a[0] += 1
    a[1] += f(r, "DEVICE FW DURATION [ns]") / 1e6
    a[2] += f(r, "OP TO OP LATENCY [ns]") / 1e6
print(f"\n{'op':<38} {'n':>5} {'FW ms':>8} {'gap ms':>8} {'gap us/op':>10}")
print("-" * 74)
for code, (n, fwm, gapm) in sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:20]:
    print(f"{code:<38} {n:>5} {fwm:>8.1f} {gapm:>8.1f} {gapm * 1e3 / n:>10.1f}")
