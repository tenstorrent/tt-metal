import json

import numpy as np
import torch

K = "/home/ttuser/atupe/qwen38_work/runs/T3/kl/"
O = "/home/ttuser/atupe/qwen38_work/runs/T3/emul/"
hf = torch.load(K + "hf38_logp.pt")
ref = torch.log_softmax(hf["logp"].float(), -1)
ram = hf["argmax"]
top5 = ref.topk(5, -1).indices
B = json.load(open(O + "bytes.json"))["GB_per_dev"]


def kl(P, Q):
    return (P.exp() * (P - Q)).sum(-1).numpy()


rows = [
    ("E1 +down BFP4", "E1-38", "E1"),
    ("E2 +attn qkvo BFP4", "E2-38", "E2"),
    ("E3 +down+attn BFP4", "E3-38", "E3"),
    ("E4 +GDN in-proj BFP4", "E4-38", "E4"),
    ("E5 +GDN out_proj BFP4", "E5-38", "E5"),
    ("E6 +lm_head BFP4", "E6-38", "E6"),
    ("E7 all BFP8 (Q8noGU)", "Q8noGU-38", "E7/Q8noGU"),
    ("E8 gate BFP4 only", "E8-38", "E8"),
    ("E9 bf16 (Q0)", "Q0-38", "E9/Q0 bf16"),
    ("ref: TT default (Q8, gu BFP4)", "Q8-38", "TT-default(Q8)"),
    ("ref: Q4 (TT Tier-2)", "Q4-38", "Q4"),
    ("ref: Q4all", "Q4all-38", "Q4all"),
]
base = None
out = []


def get(f):
    P = torch.load(O + f + "_logp.pt")["logp"]
    am = P.argmax(-1)
    e = am != ram
    return dict(
        t1=100 * (~e).float().mean().item(),
        t5=100 * (am[:, None] == top5).any(-1).float().mean().item(),
        k=kl(ref, P),
        err=e.sum().item(),
    )


b = get("Q8noGU-38")["t1"]
hdr = f"{'config':32s} {'top1%':>7s} {'d_top1_vs_E7':>12s} {'top5%':>7s} {'KLmean':>8s} {'KLmed':>8s} {'KLp99':>7s} {'errs/512':>8s} {'GB/tok/dev':>10s} {'vs TT default GB':>16s}"
out.append(hdr)
for n, f, bk in rows:
    r = get(f)
    k = r["k"]
    out.append(
        f"{n:32s} {r['t1']:7.2f} {r['t1']-b:+12.2f} {r['t5']:7.2f} {k.mean():8.4f} {np.median(k):8.4f} {np.percentile(k,99):7.3f} {r['err']:8d} {B[bk]:10.3f} {B[bk]-B['TT-default(Q8)']:+16.3f}"
    )
s = "\n".join(out)
print(s)
open(O + "results2.txt", "w").write(s + "\n")
