"""How many device ops does one Activation1d band actually cost, and what would fusing it save?

Every previous estimate of the fusion payoff was projected from op *names* rather than counted, and
today's act_block_h result showed one such projection was too optimistic. This counts instead.

The snake is exactly one op per band (TernaryDeviceOperation, 127 of them in the signposted window),
so consecutive snakes bracket one band. Counting ops and device time between them gives ops-per-band
and ms-per-band directly, from the capture already on disk -- no new profiling run.
"""

import csv
import sys
from collections import Counter

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))

start = end = None
for i, r in enumerate(rows):
    c = (r.get("OP CODE") or "").strip()
    if c == "start" and start is None:
        start = i
    elif c == "stop":
        end = i
win = rows[start + 1 : end] if start is not None and end is not None else rows

SNAKE = "TernaryDeviceOperation"
idx = [i for i, r in enumerate(win) if (r.get("OP CODE") or "").strip() == SNAKE]
print(f"window {len(win)} ops, {len(idx)} snakes (= bands)")
if len(idx) < 3:
    sys.exit("not enough snakes to bracket a band")


def ms(r):
    try:
        return float(r.get("DEVICE FW DURATION [ns]") or 0) / 1e6
    except ValueError:
        return 0.0


spans, times, comp = [], [], Counter()
for a, b in zip(idx, idx[1:]):
    n = b - a  # ops from this snake up to (not including) the next
    spans.append(n)
    times.append(sum(ms(win[j]) for j in range(a, b)))
    for j in range(a, b):
        comp[(win[j].get("OP CODE") or "?").strip()] += 1

spans_sorted = sorted(spans)
med = spans_sorted[len(spans_sorted) // 2]
med_ms = sorted(times)[len(times) // 2]
print(f"ops between consecutive snakes: median {med}, min {min(spans)}, max {max(spans)}")
print(f"device time per band:           median {med_ms:.2f} ms")

total_band_ops = sum(spans)
total_band_ms = sum(times)
print(
    f"\nall bands: {total_band_ops} ops, {total_band_ms:.1f} ms "
    f"({100.0 * total_band_ops / len(win):.0f} % of ops in window)"
)

print("\ncomposition of one median band (op -> count per band):")
for name, n in comp.most_common(12):
    print(f"  {name:<40} {n / len(spans):>6.1f}")

n_bands = len(idx)
for fused_ops in (1, 3, 5):
    saved = total_band_ops - n_bands * fused_ops
    print(
        f"\nfuse band -> {fused_ops} op(s): {total_band_ops} -> {n_bands * fused_ops} ops, "
        f"saves {saved} ops = {saved * 0.142:.0f} ms at 142 us/op"
    )
