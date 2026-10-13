# Scratch (not for merge): model_gate_cmp.py <M.pt> <N.pt>: per model config and stimulus, the gate's own checks, the PCC of
# the routing weights and of a MoE layer built on them against a float64 golden, and the expert sets of both arms.
import sys
from pathlib import Path
import torch
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig

A, B = torch.load(sys.argv[1]), torch.load(sys.argv[2])
CFG = Path("/work/models/common/modules/moe/configs")
DOUT = 16
experts = {}


def expert(name, hidden, e):
    key = (name, e)
    if key not in experts:
        g = torch.Generator().manual_seed(100003 * e + 11)
        experts[key] = torch.randn((hidden, DOUT), generator=g, dtype=torch.float64) / hidden**0.5
    return experts[key]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-300))


def layer(name, x, idx, wts):
    hidden = x.shape[-1]
    y = torch.zeros((x.shape[0], DOUT), dtype=torch.float64)
    for t in range(x.shape[0]):
        for j in range(idx.shape[1]):
            y[t] += float(wts[t, j]) * (x[t].double() @ expert(name, hidden, int(idx[t, j])))
    return y


groups = sorted({(c, s) for (c, s, _) in A})
print("GATECMP config stimulus tokens | checks M/N | weights PCC min M/N | layer PCC min M/N | layer PCC spread M/N | sets=golden M/N | M=N sets, ordered")
for c, s in groups:
    cfg = TTMoEGateConfig.from_yaml((CFG / f"{c}.yaml").read_text())
    st = {"M": dict(ok=0, wp=[], lp=[], gold=0), "N": dict(ok=0, wp=[], lp=[], gold=0)}
    same_set = same_ord = tok = 0
    for seed in sorted(sd for (cc, ss, sd) in A if cc == c and ss == s):
        ra, rb = A[(c, s, seed)], B[(c, s, seed)]
        x, w, b, pb, k, n = ra["x"], ra["w"], ra["b"], ra["pb"], ra["k"], ra["n"]
        gs, gi = TTMoEGate.golden(x, w, b, select_experts_k=k, score_func=cfg.score_func,
                                  softmax_position=cfg.softmax_position, scaling_factor=cfg.routed_scaling_factor,
                                  eps=cfg.eps, n_group=cfg.n_group, gate_proj_bias=pb)
        gi = gi.to(torch.int64)
        logits = x.float() @ w.float() + (pb.float() if pb is not None else 0)
        score = torch.sigmoid(logits) if cfg.score_func == "sigmoid" else (
            torch.sqrt(torch.nn.functional.softplus(logits)) if cfg.score_func == "sqrtsoftplus" else logits)
        bias = b.float() if b is not None else torch.zeros(n)
        key = score + bias
        gold_dense = torch.zeros((x.shape[0], n), dtype=torch.float64).scatter_(1, gi, gs.double())
        gold_layer = layer(c, x, gi, gs)
        for arm, r in (("M", ra), ("N", rb)):
            ds, di = r["ds"], r["di"]
            sel = torch.gather(score, -1, di)
            wts = torch.exp(sel) if cfg.score_func == "softmax" else sel
            exp = wts / (wts.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
            atol = 1e-2
            ok1 = torch.allclose(ds.sort(-1).values, exp.sort(-1).values, atol=atol)
            ok2 = torch.allclose(torch.gather(key, -1, di).sort(-1).values, torch.gather(key, -1, gi).sort(-1).values, atol=5e-2)
            st[arm]["ok"] += int(ok1 and ok2)
            dense = torch.zeros((x.shape[0], n), dtype=torch.float64).scatter_(1, di, ds.double())
            st[arm]["wp"].append(pcc(dense, gold_dense))
            st[arm]["lp"].append(pcc(layer(c, x, di, ds), gold_layer))
            st[arm]["gold"] += sum(set(di[t].tolist()) == set(gi[t].tolist()) for t in range(x.shape[0]))
        for t in range(x.shape[0]):
            tok += 1
            same_set += set(ra["di"][t].tolist()) == set(rb["di"][t].tolist())
            same_ord += torch.equal(ra["di"][t], rb["di"][t]) and torch.equal(ra["ds"][t], rb["ds"][t])
    m, nn = st["M"], st["N"]
    import statistics as stt
    dl = [b2 - a2 for a2, b2 in zip(m["lp"], nn["lp"])]
    dw = [b2 - a2 for a2, b2 in zip(m["wp"], nn["wp"])]
    print(f"GATESTAT {c} {s} | layer mean M {stt.mean(m['lp']):.5f} sd {stt.pstdev(m['lp']):.5f} N {stt.mean(nn['lp']):.5f} sd {stt.pstdev(nn['lp']):.5f} | paired N-M mean {stt.mean(dl):+.5f} sd {stt.pstdev(dl):.5f} N<M in {sum(d < 0 for d in dl)}/{len(dl)} | weights paired N-M mean {stt.mean(dw):+.5f}", flush=True)
    nseeds = len(m["wp"])
    f = lambda v: f"{min(v):.5f}"
    sp = lambda v: f"{max(v) - min(v):.5f}"
    print(f"GATECMP {c} {s} {tok} | {m['ok']}/{nseeds} {nn['ok']}/{nseeds} | {f(m['wp'])} {f(nn['wp'])} | {f(m['lp'])} {f(nn['lp'])} | {sp(m['lp'])} {sp(nn['lp'])} | {m['gold']} {nn['gold']} | {same_set} {same_ord}", flush=True)
