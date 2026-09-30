# t41: S2 self-attention locality from dumped Q/K/V (LTX_DUMP_QKV). CPU only.
# Usage: python analyze.py <qkv_dir> [num_queries_per_head]
# Per dump: attention mass by latent-frame distance, and the rel_l2 of the temporal-band output
# (window W frames) vs dense, both on a random query sample in fp32.
import glob
import os
import re
import sys

import torch

TPF, F = 2040, 19  # stage-2 1080p: 34x60 latent tokens per frame, 19 latent frames
WINDOWS = (1, 2, 3, 4, 6)

qkv_dir = sys.argv[1]
nq = int(sys.argv[2]) if len(sys.argv) > 2 else 512
torch.manual_seed(0)
rows = []
for qp in sorted(
    glob.glob(os.path.join(qkv_dir, "q_s*_l*.pt")), key=lambda p: tuple(map(int, re.findall(r"\d+", p)[-2:]))
):
    s, l = map(int, re.findall(r"\d+", os.path.basename(qp)))
    q = torch.load(qp).float()
    k = torch.load(qp.replace("/q_", "/k_")).float()
    v = torch.load(qp.replace("/q_", "/v_")).float()
    H, N, D = q.shape
    idx = torch.randint(0, N, (nq,))
    fq = idx // TPF
    fk = torch.arange(N) // TPF
    dist = (fq[:, None] - fk[None, :]).abs()  # (nq, N)
    logits = torch.einsum("hqd,hkd->hqk", q[:, idx], k) / D**0.5
    p = torch.softmax(logits, dim=-1)
    dense = torch.einsum("hqk,hkd->hqd", p, v)
    mass = torch.stack([(p * (dist == d)).sum(-1).mean() for d in range(F)])
    errs = {}
    for w in WINDOWS:
        pw = torch.softmax(logits.masked_fill(dist > w, float("-inf")), dim=-1)
        out = torch.einsum("hqk,hkd->hqd", pw, v)
        errs[w] = ((out - dense).norm() / dense.norm()).item()
        errs[f"in{w}"] = (p * (dist <= w)).sum(-1).mean().item()
    rows.append((s, l, mass, errs))
    print(
        f"s{s} l{l:2d} mass d0..4 "
        + " ".join(f"{m:.3f}" for m in mass[:5])
        + " | "
        + " ".join(f"W{w}: in {errs[f'in{w}']:.3f} rel_l2 {errs[w]:.4f}" for w in WINDOWS),
        flush=True,
    )

print("\nmean over dumps:")
for w in WINDOWS:
    e = [r[3][w] for r in rows]
    print(
        f"W{w}: rel_l2 mean {sum(e) / len(e):.4f} max {max(e):.4f}  mass-in mean {sum(r[3][f'in{w}'] for r in rows) / len(rows):.3f}"
    )
