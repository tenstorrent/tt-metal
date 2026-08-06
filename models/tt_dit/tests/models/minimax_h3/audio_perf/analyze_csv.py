"""Group the Tracy op CSV by op code within the signposted window: count, device FW ms, share."""

import csv
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))

# The signposts appear as their own rows in the op stream; keep only what lies between them.
start = end = None
for i, r in enumerate(rows):
    code = (r.get("OP CODE") or "").strip()
    if code == "start" and start is None:
        start = i
    elif code == "stop":
        end = i
window = rows[start + 1 : end] if start is not None and end is not None else rows
print(f"rows total {len(rows)}, signpost window {len(window)} (start={start}, stop={end})")

agg = defaultdict(lambda: [0, 0.0])
for r in window:
    code = (r.get("OP CODE") or "?").strip()
    try:
        ns = float(r.get("DEVICE FW DURATION [ns]") or 0)
    except ValueError:
        ns = 0.0
    agg[code][0] += 1
    agg[code][1] += ns / 1e6

total = sum(v[1] for v in agg.values())
print(f"\n{'op':<40} {'n':>6} {'ms':>9} {'%':>6}")
print("-" * 66)
for code, (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:22]:
    print(f"{code:<40} {n:>6} {ms:>9.1f} {100.0 * ms / max(total, 1e-9):>6.1f}")
print(f"\ntotal device FW in window: {total:.1f} ms over {sum(v[0] for v in agg.values())} ops")
