# Per-stage gain of the first transformer block of up_blocks.0 from SDXL_UPBLOCK_DUMP dumps (tt vs torch).
import os
import sys

import torch

d = sys.argv[1]
for nm in ("ln1", "attn1", "add1", "ln2", "attn2", "add2", "ln3", "ff", "add3"):
    tt = torch.load(f"{d}/tb_{nm}.pt").float().flatten()
    tnm = {"ln1": "norm1", "ln2": "norm2", "ln3": "norm3"}.get(nm, nm)
    tp = f"{d}/torch_tb_{tnm}.pt"
    if not os.path.exists(tp):
        print(f"{nm:6s} tt_std={tt.std():.4f} (no torch ref)")
        continue
    ref = torch.load(tp).float().flatten()
    if ref.numel() != tt.numel():
        print(f"{nm:6s} shape mismatch {ref.numel()} vs {tt.numel()}")
        continue
    gain = (tt * ref).sum() / (ref * ref).sum()
    pcc = torch.corrcoef(torch.stack([tt, ref]))[0, 1]
    print(
        f"{nm:6s} gain={gain:.4f} std_ratio={tt.std()/ref.std():.4f} bias={(tt-ref).mean():+.5f} rel_rms={((tt-ref).pow(2).mean().sqrt()/ref.std()):.4f} pcc={pcc:.5f}"
    )
