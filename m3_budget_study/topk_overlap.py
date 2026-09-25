#!/usr/bin/env python3
"""Indexer top-k agreement between a packed and a reference budget_packed.py BUDGET_TOPK_DUMP capture.

  topk_overlap.py <reference.pt> <packed.pt>

For every (sparse layer, segment): the fraction of the k selected blocks that agree, per query row and
group, averaged over the segment's real rows. Packed captures are layer-major (layer, then segment);
reference captures run each segment alone, so they are segment-major.
"""
import sys

import torch


def overlap(a, b):
    k = a.shape[-1]
    return (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum(-1).float() / k


def main(ref_path, packed_path):
    R, P = torch.load(ref_path), torch.load(packed_path)
    segs = P["segments"]
    S = len(segs)
    L = len(P["ids"]) // S
    assert len(R["ids"]) == L * S, (len(R["ids"]), L, S)
    worst = 1.0
    for s, (h, n) in enumerate(segs):
        vals = []
        for l in range(L):
            a = R["ids"][s * L + l].long()[..., :n, :]
            b = P["ids"][l * S + s].long()[..., :n, :]
            o = overlap(a, b)
            vals.append((float(o.mean()), float((o == 1).float().mean())))
            worst = min(worst, vals[-1][0])
        print(
            f"segment {s} (h={h}, n={n}): mean overlap per sparse layer "
            + " ".join(f"{m:.4f}" for m, _ in vals)
            + " | rows identical "
            + " ".join(f"{e:.3f}" for _, e in vals)
        )
    print(f"worst mean overlap {worst:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
