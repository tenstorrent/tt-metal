"""CPU: reference logits for the real dump states (layer-24 iHC streams of the 8 oracle prompts) -> head_ref.pt (hidden bf16, top1/top2/margin of all rows, full fp32 logits of a subset).
Also: how much do norm-order choices (fused vs two-rounding RMSNorm) change top-1 on real rows (a property of the reference, not of the device).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt"))
from hy4_head_ref import *

D = "/mnt/tt-data/ssinghal/hy4-logs/program/lm-head"
W = torch.load(f"{D}/head_weights.pt")
W["lm_head"] = W["lm_head"].float()
st = torch.load("/mnt/tt-data/ssinghal/hy4-logs/ref_dumps/dump_v1/state.pt")
X = torch.cat([x[0] for x in st["x4s"]])  # [N,4,6144] fp32
print("rows", X.shape, "layer", st["layer"])
Xb = X.bfloat16()  # vllm streams are bf16 between layers
H = head_hidden(Xb, W)
H2 = head_hidden(Xb, W, norm="two")
t1 = t2 = None
top1, top2, marg = [], [], []
keep = {}
SUB = list(range(0, X.shape[0], 7))[:512]
for i in range(0, X.shape[0], 128):
    L = logits_ref(H[i : i + 128], W["lm_head"])
    v, ix = L.topk(2, dim=-1)
    top1.append(ix[:, 0])
    top2.append(ix[:, 1])
    marg.append(v[:, 0] - v[:, 1])
    for j in range(L.shape[0]):
        if i + j in SUB:
            keep[i + j] = L[j].clone()
    print(i, flush=True)
top1 = torch.cat(top1)
marg = torch.cat(marg)
L2 = torch.cat([logits_ref(H2[i : i + 128], W["lm_head"]).argmax(-1) for i in range(0, X.shape[0], 128)])
print("norm-order top1 agreement (fused vs two-rounding):", float((L2 == top1).float().mean()))
print(
    "margin quantiles",
    [float(marg.quantile(q)) for q in (0.001, 0.01, 0.05, 0.5)],
    "frac margin<0.01:",
    float((marg < 0.01).float().mean()),
    "<0.05",
    float((marg < 0.05).float().mean()),
)
torch.save(
    {"X": Xb, "H": H, "top1": top1, "marg": marg, "sub": SUB, "sub_logits": torch.stack([keep[i] for i in SUB])},
    f"{D}/head_ref.pt",
)
