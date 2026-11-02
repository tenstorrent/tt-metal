#!/usr/bin/env python3
"""Compare two BONSAI_E2E_DUMP files position by position against a reference: where do the configs' argmaxes
differ, how confident was each (top-1/top-2 logit margin), and which one matched the reference there.

Near-tie flips (small margins in both arms) are numerics noise; flips where an arm was confident and wrong point at
a real error. Usage: flip_analysis.py <dump_a.pt> <dump_b.pt> [ref.refpt] [--scored]"""
import sys

import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
ref = torch.load(sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else a["reference"])
top5 = ref["top5_tokens"]
n = len(a["preds"])
span = range(ref["reference_tokens"].shape[1] // 2 - 1, n) if "--scored" in sys.argv else range(n)
diff = [i for i in span if a["preds"][i] != b["preds"][i]]
print(f"{len(diff)} positions differ of {len(span)}; reference {sys.argv[3] if len(sys.argv) > 3 else a['reference']}")
print(f"{'pos':>5} {'A tok':>8} {'A marg':>7} {'A rank':>6} {'B tok':>8} {'B marg':>7} {'B rank':>6}")
win = {"A": 0, "B": 0}
for i in diff:
    t5 = top5[i].tolist()
    ra = t5.index(a["preds"][i]) if a["preds"][i] in t5 else -1
    rb = t5.index(b["preds"][i]) if b["preds"][i] in t5 else -1
    win["A"] += ra == 0
    win["B"] += rb == 0
    print(f"{i:5d} {a['preds'][i]:8d} {a['margins'][i]:7.3f} {ra:6d} {b['preds'][i]:8d} {b['margins'][i]:7.3f} {rb:6d}")
ma = [a["margins"][i] for i in diff]
mb = [b["margins"][i] for i in diff]
all_m = sorted(a["margins"][i] for i in span)
print(f"top-1 matches among the differing positions: A {win['A']}, B {win['B']}")
if diff:
    print(f"margins at the differing positions: A median {sorted(ma)[len(ma)//2]:.3f} max {max(ma):.3f}; "
          f"B median {sorted(mb)[len(mb)//2]:.3f} max {max(mb):.3f}; all positions' median margin {all_m[len(all_m)//2]:.3f}")
