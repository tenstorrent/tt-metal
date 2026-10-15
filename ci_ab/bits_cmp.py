#!/usr/bin/env python3
"""Scratch, not for merge. Per case: are A's and B's first outputs bit identical; if not, PCC and max abs difference.
B2 against B checks determinism."""
import os
import sys

import torch

root = sys.argv[1]


def load(arm, name):
    p = os.path.join(root, f"bits_{arm}", name)
    return torch.load(p, weights_only=True) if os.path.exists(p) else None


def cmp(x, y):
    if x is None or y is None:
        return "missing"
    if len(x) != len(y) or any(a.shape != b.shape for a, b in zip(x, y)):
        return "shape"
    if all(torch.equal(a, b) for a, b in zip(x, y)):
        return "equal"
    a = torch.cat([t.float().flatten() for t in x])
    b = torch.cat([t.float().flatten() for t in y])
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return f"differ pcc={pcc:.6f} maxabs={(a - b).abs().max().item():.4g}"


for name in sorted(n for n in os.listdir(os.path.join(root, "bits_B")) if n.endswith(".pt")):
    print(f"TAB BITS {name[:-3]} AvsB {cmp(load('A', name), load('B', name))} B2vsB {cmp(load('B2', name), load('B', name))}")
