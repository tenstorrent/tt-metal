#!/usr/bin/env python3
"""Scratch, not for merge. Flux joint cases: error of A's and B's first output against an fp64 torch reference built
from the same seeded inputs (non causal attention over the spatial then joint tokens, default scale). Prints rmse and
max abs error per output (spatial, joint) and over both, and checks that the regenerated inputs hash like the job's."""
import hashlib
import math
import os
import sys

import torch

root = sys.argv[1]
N, L, D = 4096, 512, 128
CASES = (("flux_joint_H6_N4096_L512", 6), ("flux_joint_H12_N4096_L512", 12), ("flux_joint_H24_N4096_L512", 24),
         ("flux_joint_H24_N4096_L512_approx", 24))
torch.set_num_threads(os.cpu_count() or 1)
refs = {}


def reference(H):
    torch.manual_seed(0)
    xs = [torch.randn((1, H, n, D), dtype=torch.bfloat16) for n in (N, N, N, L, L, L)]
    md5 = hashlib.md5(b"".join(x.view(torch.int16).numpy().tobytes() for x in xs)).hexdigest()[:12]
    q, k, v = (torch.cat([xs[i], xs[i + 3]], dim=2)[0].double() for i in range(3))
    out = torch.empty(H, N + L, D, dtype=torch.float64)
    for h in range(H):
        out[h] = torch.softmax((q[h] @ k[h].T) / math.sqrt(D), dim=-1) @ v[h]
    return md5, out


def err(o, ref):
    e = o - ref
    return e.pow(2).mean().sqrt().item(), e.abs().max().item()


print("TAB REFHDR case inputs_md5 saved_md5 arm rmse_all max_all rmse_spatial rmse_joint ref_rms")
for name, H in CASES:
    if H not in refs:
        refs[H] = reference(H)
    md5, ref = refs[H]
    p = os.path.join(root, "bits_B", f"{name}.inputs")
    saved = open(p).read().strip() if os.path.exists(p) else "missing"
    ref_rms = ref.pow(2).mean().sqrt().item()
    for arm in ("A", "B", "B2"):
        p = os.path.join(root, f"bits_{arm}", f"{name}.pt")
        if not os.path.exists(p):
            print(f"TAB REF {name} {md5} {saved} {arm} missing")
            continue
        outs = torch.load(p, weights_only=True)
        o = torch.cat([outs[0][0, :, :N].double(), outs[1][0, :, :L].double()], dim=1)
        ra, ma = err(o, ref)
        rs, _ = err(o[:, :N], ref[:, :N])
        rj, _ = err(o[:, N:], ref[:, N:])
        print(f"TAB REF {name} {md5} {saved} {arm} {ra:.6e} {ma:.6e} {rs:.6e} {rj:.6e} {ref_rms:.6e}", flush=True)
