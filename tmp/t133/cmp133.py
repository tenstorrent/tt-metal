"""#133 A/B summary, CPU only: traced/eager decode min per arm from run133.log, and A vs B yuv bit-identity.

Usage (on blx03 after the job): python tmp/t133/cmp133.py /var/tmp/fasth3/t133
"""

import re
import sys
from pathlib import Path

import torch

V = Path(sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/fasth3/t133")
log = (V / "run133.log").read_text()
tag = None
mins = {}
for line in log.splitlines():
    m = re.match(r"\[t133\] arm=(\w) tag=(\w+)", line)
    if m:
        tag = m.group(2)
    m = re.match(r"AB arm=(eager|traced) decode_s=.* min=([\d.]+)", line)
    if m and tag and tag != "ref":
        mins.setdefault((tag[0], m.group(1)), []).append(float(m.group(2)))
for mode in ("eager", "traced"):
    a, b = mins.get(("A", mode), []), mins.get(("B", mode), [])
    if a and b:
        d = min(b) - min(a)
        print(
            f"T133_CMP {mode} A_min={min(a):.4f} B_min={min(b):.4f} delta_ms={d * 1e3:+.1f} ({d / min(a) * 100:+.2f}%) A={a} B={b}"
        )
outs = sorted(p for p in V.glob("out_[AB]*/yuv_traced.pt"))
ref = next((p for p in outs if p.parent.name.startswith("out_A")), None)
for p in outs:
    if ref is not None and p != ref:
        x, y = torch.load(ref), torch.load(p)
        same = x.shape == y.shape and torch.equal(x, y)
        diff = (x.int() - y.int()).abs().max().item() if x.shape == y.shape else "shape"
        print(f"T133_CMP {p.parent.name}_vs_{ref.parent.name} identical={same} max_abs_diff={diff}")
