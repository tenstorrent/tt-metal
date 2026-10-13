#!/usr/bin/env python3
# Compare the float column SUM/AVG row-0 words of two FCOL_OUT dirs (main against a candidate). usage: fcol_cmp.py <A> <B>
import glob, os, sys, torch
a, b = sys.argv[1:3]; n = same = files = 0; bad = []
for fa in sorted(glob.glob(f"{a}/*.pt")):
    fb = f"{b}/{os.path.basename(fa)}"
    if not os.path.exists(fb): bad.append(f"missing {os.path.basename(fa)}"); continue
    x, y = torch.load(fa), torch.load(fb); files += 1
    n += x.numel(); same += int((x == y).sum())
    if not torch.equal(x, y): bad.append(f"{os.path.basename(fa)}: {int((x != y).sum())} of {x.numel()} words differ")
print(f"FCOL {files} cases, {same} of {n} row-0 words identical"); print("\n".join(bad[:20]))
