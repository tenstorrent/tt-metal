# SPDX-License-Identifier: Apache-2.0
"""Compare a tagged main_pcc run against the moe baseline reference.

Gate for numerically-equivalent placement/CCL-fusion changes:
  - logits (to_layout_267) PCC vs baseline ~1.0
  - argmax next-token agreement == 100%
Usage: python3 compare_rfuse.py <new_tag> [base_tag=moe]
"""
import os, sys, torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_io")
NEW = sys.argv[1] if len(sys.argv) > 1 else "rfuse"
BASE = sys.argv[2] if len(sys.argv) > 2 else "moe"


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


A = torch.load(os.path.join(OUT, f"main_outs_{BASE}.pt"))
B = torch.load(os.path.join(OUT, f"main_outs_{NEW}.pt"))
print(f"=== compare new='{NEW}' vs base='{BASE}' ===", flush=True)
worst = 1.0
for n in A:
    p = _pcc(A[n], B[n])
    worst = min(worst, p)
    print(
        f"  {n:16s} PCC={p:.6f} base_norm={float(A[n].norm()):.3f} new_norm={float(B[n].norm()):.3f} shape={tuple(A[n].shape)}",
        flush=True,
    )
print(f"=== WORST live-out PCC = {worst:.6f} ===", flush=True)

# argmax token agreement on lm_head logits (to_layout_267)
lg = "to_layout_267"
if lg in A and lg in B:
    # shape [num_dev, 32tok, 129280]; argmax over vocab, compare per (dev,tok)
    am_a = A[lg].argmax(dim=-1)
    am_b = B[lg].argmax(dim=-1)
    agree = (am_a == am_b).float().mean().item()
    print(f"=== argmax token agreement = {agree*100:.4f}% ({int((am_a==am_b).sum())}/{am_a.numel()}) ===", flush=True)
    print(
        f"=== GATE: {'PASS' if (worst >= 0.99 and agree == 1.0) else 'FAIL'} (PCC>=0.99 AND argmax 100%) ===",
        flush=True,
    )
