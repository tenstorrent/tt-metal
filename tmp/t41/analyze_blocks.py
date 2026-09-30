# t41: block-sparse top-k S2 self-attention at ring-joint chunk granularity (q192/k512, contiguous tokens,
# no reordering). For sampled query chunks, keep the top fraction of key chunks, chosen either by the true
# attention mass (oracle) or by mean-pooled q.k scores (cheap estimate), and report rel_l2 vs dense. CPU only.
# Usage: python analyze_blocks.py <qkv_dir> [num_query_chunks]
import glob
import os
import re
import sys

import torch

QC, KC = 192, 512
KEEP = (0.25, 0.5, 0.75)

qkv_dir = sys.argv[1]
nqc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
torch.manual_seed(0)
acc = {}
for qp in sorted(
    glob.glob(os.path.join(qkv_dir, "q_s*_l*.pt")), key=lambda p: tuple(map(int, re.findall(r"\d+", p)[-2:]))
):
    s, l = map(int, re.findall(r"\d+", os.path.basename(qp)))
    q = torch.load(qp).float()
    k = torch.load(qp.replace("/q_", "/k_")).float()
    v = torch.load(qp.replace("/q_", "/v_")).float()
    H, N, D = q.shape
    nk = (N + KC - 1) // KC
    kpad = torch.nn.functional.pad(k, (0, 0, 0, nk * KC - N))
    kmean = kpad.view(H, nk, KC, D).sum(2) / torch.tensor([min(KC, N - i * KC) for i in range(nk)])[None, :, None]
    kchunk = torch.arange(N) // KC
    line = []
    for qc in torch.randint(0, N // QC, (nqc,)).tolist():
        qs = q[:, qc * QC : (qc + 1) * QC]
        logits = torch.einsum("hqd,hkd->hqk", qs, k) / D**0.5
        p = torch.softmax(logits, dim=-1)
        dense = torch.einsum("hqk,hkd->hqd", p, v)
        true_mass = torch.zeros(H, nk).index_add_(1, kchunk, p.sum(1))
        est = torch.einsum("hd,hkd->hk", qs.mean(1), kmean)
        for keep in KEEP:
            n = max(1, round(keep * nk))
            for name, score in (("oracle", true_mass), ("pooled", est)):
                sel = torch.zeros(H, nk, dtype=torch.bool).scatter_(1, score.topk(n, dim=1).indices, True)
                pw = torch.softmax(logits.masked_fill(~sel[:, None, kchunk], float("-inf")), dim=-1)
                out = torch.einsum("hqk,hkd->hqd", pw, v)
                acc.setdefault((name, keep), []).append(((out - dense).norm() / dense.norm()).item())
                acc.setdefault((name, keep, "l"), {}).setdefault(l, []).append(acc[(name, keep)][-1])
    print(
        f"s{s} l{l:2d} "
        + " ".join(f"{n}{k:.2f} {sum(acc[(n, k)][-nqc:]) / nqc:.3f}" for n in ("oracle", "pooled") for k in KEEP),
        flush=True,
    )

print("\nmean rel_l2 over dumps:")
for name in ("oracle", "pooled"):
    for keep in KEEP:
        e = acc[(name, keep)]
        print(f"{name} keep {keep:.2f}: mean {sum(e) / len(e):.4f} max {max(e):.4f}")
