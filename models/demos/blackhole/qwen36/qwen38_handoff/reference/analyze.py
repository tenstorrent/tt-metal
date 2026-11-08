import numpy as np
import torch
from transformers import AutoTokenizer

O = "/home/ttuser/atupe/qwen38_work/runs/T3/kl/"
R = "models/tt_transformers/tests/reference_outputs/"
tok = AutoTokenizer.from_pretrained("/home/ttuser/atupe/models/Qwen3.8-27B")
hf = {k: torch.load(O + f"hf{k}_logp.pt") for k in ("36", "38")}
tt = {k: torch.load(O + f"tt{k}.pt") for k in ("36", "38")}
ref = {k: torch.load(R + f"Qwen3.{6 if k=='36' else 8}-27B.refpt")["top5_tokens"][511:1023, 0] for k in ("36", "38")}
for k in hf:
    print(
        k,
        "hf argmax==refpt top1:",
        (hf[k]["argmax"] == ref[k]).float().mean().item(),
        "tt argmax==refpt (acc):",
        (tt[k]["argmax"] == ref[k]).float().mean().item(),
    )
print(
    "tokens-fed equal 36/38 refpt? ",
    torch.equal(
        torch.load(R + "Qwen3.6-27B.refpt")["reference_tokens"][0, 512:1024],
        torch.load(R + "Qwen3.8-27B.refpt")["reference_tokens"][0, 512:1024],
    ),
)


def lp(x):
    return torch.log_softmax(x.float(), -1)


def kl(P, Q):  # P,Q logp [N,V]
    return (P.exp() * (P - Q)).sum(-1)


def pcc(a, b):
    a = a.float() - a.float().mean(-1, keepdim=True)
    b = b.float() - b.float().mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def st(x):
    x = np.asarray(x)
    return f"mean {x.mean():.4f} med {np.median(x):.4f} p90 {np.percentile(x,90):.4f} p99 {np.percentile(x,99):.4f}"


TT = {k: lp(tt[k]["logits"]) for k in tt}
HF = {k: hf[k]["logp"].float() for k in hf}
HFl = {k: lp(hf[k]["logp"]) for k in hf}  # renormalise fp16
res = {}
for k in ("36", "38"):
    P = HFl[k]
    Q = TT[k]
    K = kl(P, Q).numpy()
    pc = pcc(hf[k]["logp"], tt[k]["logits"]).numpy()
    dis = (tt[k]["argmax"] != hf[k]["argmax"]).numpy()
    top2 = P.topk(2, -1).values
    marg = (top2[:, 0] - top2[:, 1]).numpy()
    res[k] = (K, pc, dis, marg)
    print(f"\n== {k} a. top1 agree {100*(1-dis.mean()):.2f}% ({dis.sum()} disagreements)")
    print(" b. KL", st(K), " | PCC mean %.5f min %.5f" % (pc.mean(), pc.min()))
    print(
        " c. margin median %.3f ; <0.1 %.3f <0.5 %.3f <1.0 %.3f"
        % (np.median(marg), (marg < 0.1).mean(), (marg < 0.5).mean(), (marg < 1).mean())
    )
    print(" d. bins:")
    for lo, hi in [(0, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 1e9)]:
        m = (marg >= lo) & (marg < hi)
        print(
            f"   [{lo},{hi}) n={m.sum():3d} dis={dis[m].sum():3d} rate={dis[m].mean() if m.sum() else 0:.3f} meanKL={K[m].mean() if m.sum() else 0:.4f}"
        )
    print(" f. blocks:")
    for b in range(4):
        s = slice(b * 128, (b + 1) * 128)
        print(f"   {b*128}-{b*128+127}: dis {dis[s].sum()}/128 meanKL {K[s].mean():.4f} medKL {np.median(K[s]):.4f}")
# e leak
print("\n== e. leak")
for name, Q, P, ta, ha in [
    ("TT38 vs HF38", TT["38"], HFl["38"], tt["38"]["argmax"], hf["38"]["argmax"]),
    ("TT38 vs HF36", TT["38"], HFl["36"], tt["38"]["argmax"], hf["36"]["argmax"]),
    ("TT36 vs HF36", TT["36"], HFl["36"], tt["36"]["argmax"], hf["36"]["argmax"]),
    ("TT36 vs HF38", TT["36"], HFl["38"], tt["36"]["argmax"], hf["38"]["argmax"]),
    ("HF36 vs HF38 (argmax; KL(HF38||HF36))", HFl["36"], HFl["38"], hf["36"]["argmax"], hf["38"]["argmax"]),
    ("TT36 vs TT38 (argmax; KL(TT38||TT36))", TT["36"], TT["38"], tt["36"]["argmax"], tt["38"]["argmax"]),
]:
    K = kl(P, Q).numpy()
    print(f" {name}: agree {100*(ta==ha).float().mean():.2f}%  KL {st(K)}")
# dis overlap
d38 = res["38"][2]
d36 = res["36"][2]
h = (hf["36"]["argmax"] != hf["38"]["argmax"]).numpy()
print(
    " disagreements TT38 on positions where HF36!=HF38:",
    d38[h].sum(),
    "of",
    h.sum(),
    "; on other positions:",
    d38[~h].sum(),
    "of",
    (~h).sum(),
)
print(
    " TT38==HF36 where TT38!=HF38:", ((tt["38"]["argmax"] == hf["36"]["argmax"]).numpy() & d38).sum(), "of", d38.sum()
)
# g worst
K = res["38"][0]
idx = np.argsort(-K)[:5]
for i in idx:
    print(
        f"\n pos {i} (refpt row {511+i}) KL {K[i]:.3f} margin {res['38'][3][i]:.3f}  fed/next-ref-token={tok.decode([int(ref['38'][i])])!r}"
    )
    for nm, L in (("HF38", HFl["38"]), ("TT38", TT["38"]), ("HF36", HFl["36"])):
        v, t = L[i].topk(3)
        print("   ", nm, [(tok.decode([int(a)]), int(a), round(float(b), 3)) for b, a in zip(v, t)])
