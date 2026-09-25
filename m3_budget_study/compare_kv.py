#!/usr/bin/env python3
"""PCC of per-slot K / V / index_k between two budget_packed.py KV dumps, over real positions only.

  compare_kv.py <dirA> <dirB> slot:real_len [slot:real_len ...]      (sp = 4, segment = 2048)

The dumps are in the raw on-device layout (block-cyclic, period one 2048-token segment): position p lives
on SP chip (p % 2048) // 512, local row (p // 2048) * 512 + p % 512, i.e. composed row chip * cap/sp + local.
"""
import sys

import torch

SP, SEG = 4, 2048


def rows_for(real_len, cap):
    p = torch.arange(real_len)
    return ((p % SEG) // (SEG // SP)) * (cap // SP) + (p // SEG) * (SEG // SP) + p % (SEG // SP)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main(da, db, *specs):
    worst = 1.0
    for spec in specs:
        slot, real = map(int, spec.split(":"))
        A, Bt = torch.load(f"{da}/slot{slot}.pt"), torch.load(f"{db}/slot{slot}.pt")
        for name, a, b in zip(("k", "v", "index_k"), A, Bt):
            idx = rows_for(real, a.shape[2])
            per_layer = [pcc(a[l][:, idx], b[l][:, idx]) for l in range(a.shape[0])]
            if name == "index_k":  # dense layers never write index_k
                per_layer = [p for l, p in enumerate(per_layer) if a[l][:, idx].abs().sum() > 0]
            worst = min([worst] + per_layer)
            print(f"slot {slot} {name:8s} real={real}: per-layer PCC " + " ".join(f"{p:.5f}" for p in per_layer))
    print(f"worst PCC {worst:.5f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
