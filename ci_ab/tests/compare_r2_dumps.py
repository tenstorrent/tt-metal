#!/usr/bin/env python3
"""(sfpu_unary round 2) Compare two directories of bit dumps (test_sfpu_unary_r2_dump.py; int16 tensors of bf16 outputs,
int32 tensors of fp32 outputs): per file, the elements whose bits differ, split into non-NaN differences (a numerics
change) and NaN-payload-only differences. Optional third argument: a comma-separated list of op names to compare.
usage: compare_r2_dumps.py <dir_before> <dir_after> [ops]"""
import os
import sys

import torch

a, b = sys.argv[1], sys.argv[2]
ops = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
worst, n = 0, 0
for f in sorted(os.listdir(a)):
    if not f.endswith(".pt"):
        continue
    if ops is not None and f.rsplit("_", 1)[0] not in ops:
        continue
    if not os.path.exists(os.path.join(b, f)):
        print(f"{f}: missing on one side")
        worst = 1
        continue
    x, y = torch.load(os.path.join(a, f), weights_only=True), torch.load(os.path.join(b, f), weights_only=True)
    if x.dtype in (torch.int16, torch.int32):
        fl = torch.bfloat16 if x.dtype == torch.int16 else torch.float32
        nan_both = torch.isnan(x.view(fl)) & torch.isnan(y.view(fl))
    else:  # unsigned outputs (uint16): no NaN
        nan_both = torch.zeros_like(x, dtype=torch.bool)
    diff = x != y
    non_nan, nan_only = int((diff & ~nan_both).sum()), int((diff & nan_both).sum())
    state = "identical" if non_nan == 0 and nan_only == 0 else "DIFFERS"
    print(f"{f[:-3]}: {state}; differing non-NaN elements {non_nan} of {x.numel()}; NaN payload only {nan_only}")
    worst = max(worst, 1 if non_nan else 0)
    n += 1
print(f"{n} files compared")
sys.exit(worst)
