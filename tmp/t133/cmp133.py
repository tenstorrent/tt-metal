"""#133 A/B summary, CPU only: eager/traced decode times per arm and job from run133_<job>.log, the A-vs-A and
B-vs-B spread across jobs (noise), and bit-identity of every saved traced yuv against the first A output.

Usage (on blx03 after the jobs): python tmp/t133/cmp133.py /var/tmp/fasth3/t133
"""

import re
import statistics
import sys
from pathlib import Path

import torch

V = Path(sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/fasth3/t133")
runs = {}  # (arm, mode) -> list of (tag, min, med, samples)
for log in sorted(V.glob("run133_*.log")):
    tag = None
    for line in log.read_text().splitlines():
        m = re.match(r"\[t133\] arm=(\w) tag=(\w+)", line)
        if m:
            tag = m.group(2)
        m = re.match(r"AB arm=(eager|traced) decode_s=([\d. ]+) min=", line)
        if m and tag:
            ts = [float(t) for t in m.group(2).split()]
            runs.setdefault((tag[0], m.group(1)), []).append((tag, min(ts), statistics.median(ts), ts))
for mode in ("eager", "traced"):
    for arm in "AB":
        for tag, mn, md, ts in runs.get((arm, mode), []):
            print(f"T133_RUN {mode} {tag} min={mn * 1e3:.1f}ms med={md * 1e3:.1f}ms n={len(ts)}")
    a, b = runs.get(("A", mode), []), runs.get(("B", mode), [])
    if a and b:
        am, bm = statistics.mean(r[2] for r in a), statistics.mean(r[2] for r in b)
        an, bn = [r[2] for r in a], [r[2] for r in b]
        print(
            f"T133_CMP {mode} A_med={am * 1e3:.1f}ms B_med={bm * 1e3:.1f}ms delta={(bm - am) * 1e3:+.1f}ms "
            f"({(bm - am) / am * 100:+.2f}%) A_min={min(r[1] for r in a) * 1e3:.1f} B_min={min(r[1] for r in b) * 1e3:.1f} "
            f"noise A-vs-A={(max(an) - min(an)) * 1e3:.1f}ms B-vs-B={(max(bn) - min(bn)) * 1e3:.1f}ms"
        )
outs = sorted(V.glob("out_[AB]*/yuv_traced.pt"))
ref = next((p for p in outs if p.parent.name.startswith("out_A")), None)
for p in outs:
    if ref is not None and p != ref:
        x, y = torch.load(ref), torch.load(p)
        same = x.shape == y.shape and torch.equal(x, y)
        diff = (x.int() - y.int()).abs().max().item() if x.shape == y.shape else "shape"
        print(f"T133_CMP {p.parent.name}_vs_{ref.parent.name} identical={same} max_abs_diff={diff}")
