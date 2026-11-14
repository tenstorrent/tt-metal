import os

import numpy as np
import torch

K = "/home/ttuser/atupe/qwen38_work/runs/T3/kl/"
O = "/home/ttuser/atupe/qwen38_work/runs/T3/emul/"


def lp(x):
    return torch.log_softmax(x.float(), -1)


def kl(P, Q):
    return (P.exp() * (P - Q)).sum(-1).numpy()


def st(x):
    return f"{x.mean():.4f}/{np.median(x):.4f}/{np.percentile(x,99):.3f}"


hf = {t: torch.load(K + f"hf{t}_logp.pt") for t in ("36", "38")}
tt = {t: torch.load(K + f"tt{t}.pt") for t in ("36", "38")}
HF = {t: lp(hf[t]["logp"]) for t in hf}
TT = {t: lp(tt[t]["logits"]) for t in tt}
print(
    "ref row: TT-38 vs HF38: top1 %.2f%% KL %s ; TT-36 vs HF36: %.2f%% KL %s"
    % (
        100 * (tt["38"]["argmax"] == hf["38"]["argmax"]).float().mean(),
        st(kl(HF["38"], TT["38"])),
        100 * (tt["36"]["argmax"] == hf["36"]["argmax"]).float().mean(),
        st(kl(HF["36"], TT["36"])),
    )
)
print("KL cols are mean/median/p99")
for f in sorted(os.listdir(O)):
    if not f.endswith("_logp.pt"):
        continue
    name = f[:-8]
    t = name.split("-")[1]
    E = torch.load(O + f)
    P = E["logp"]
    am = P.argmax(-1)
    ref = HF[t]
    ram = hf[t]["argmax"]
    ta = tt[t]["argmax"]
    top5 = ref.topk(5, -1).indices
    err = am != ram
    terr = ta != ram
    inter = (err & terr).sum().item()
    uni = (err | terr).sum().item()
    print(
        f"{name}: top1 vs HFbf16 {100*(~err).float().mean():.2f}% ({err.sum().item()} dis) | top5 {100*(am[:,None]==top5).any(-1).float().mean():.2f}% | KL(HF||emul) {st(kl(ref,P))} | "
        f"argmax==TT {100*(am==ta).float().mean():.2f}% | KL(emul||TT) mean {kl(P,TT[t]).mean():.4f} | err Jaccard w/ TT {inter}/{uni}={inter/max(uni,1):.3f} (emul errs {err.sum().item()}, TT errs {terr.sum().item()})"
    )
