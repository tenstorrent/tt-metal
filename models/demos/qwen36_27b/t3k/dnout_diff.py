# SPDX-License-Identifier: Apache-2.0
"""Compare TT DeltaNet block output (pre-residual, out_proj result) vs HF oracle dnout.
recur state already matches (recurrence correct) -> if dnout is off, the bug is in the
output path: q@S, gated RMSNorm, or out_proj. Per-position breakdown included."""
import torch

hf = torch.load("/home/yito/work/hf_ref_states.pt", map_location="cpu")


def pcc(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    if a.shape != b.shape:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


print("=== DeltaNet block output (pre-residual) PCC: TT vs HF dnout ===")
for li in [0, 1, 2, 4, 5, 6]:
    try:
        tt = torch.load(f"/home/yito/work/tt_dnout_{li}.pt", map_location="cpu")  # [1,S,5120]
    except FileNotFoundError:
        print(f"  layer{li}: tt_dnout missing"); continue
    h = hf.get(f"dnout{li}")
    if h is None:
        print(f"  layer{li}: hf dnout missing"); continue
    overall = pcc(tt, h)
    nratio = float(tt.float().norm() / h.float().norm())
    S = tt.shape[1]
    pp = [pcc(tt[0, p], h[0, p]) for p in range(S)]
    print(f"  layer{li}: overall PCC={overall:.5f}  norm(TT/HF)={nratio:.4f}  "
          f"per-pos={['%.4f'%x for x in pp]}")
