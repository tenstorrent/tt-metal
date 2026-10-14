#!/usr/bin/env python3
"""Scratch, not for merge. tli_cmp.py <a.pt> <b.pt>: per case, are the index tensors equal, and do both match torch.topk's
value set (the op returns indices; equal indices are the bit check)."""
import sys
import torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
same = 0
for name in a:
    eq = name in b and torch.equal(a[name], b[name])
    same += eq
    print(f"TLI CMP {name} {'identical' if eq else 'DIFFER'}")
print(f"TLI CMP total {same} of {len(a)} identical")
