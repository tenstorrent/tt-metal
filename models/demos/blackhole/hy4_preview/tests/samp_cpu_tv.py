"""CPU exactness study of the sampling LAWS on real Hy4 final-layer logits (lm-head/final_ref.pt: 3385 rows x 120832 fp32 from the oracle final states).
Compares, per row, the exact law (fp32 full vocabulary softmax(l/T) with top-k / top-p) against
  shared : models/common/sampling TTSampling as configured by format_sampling_params: logits cast to bf16, per-column top-32 candidates (8 columns), k clamped to [1,32] (off -> 32),
           inverse temperature stored as bf16, top-p / temperature on the 32 candidates (law emulation; the ttnn.sampling kernel itself is not run here)
  shared32: same but candidate values kept in fp32 (isolates the top-32 truncation from the bf16 rounding)
  thr     : DSV4-style full-vocabulary threshold sampler (emulate_keep_weights, J=32 levels=4): the law the hybrid would use for 'no top-k'
Usage: python samp_cpu_tv.py [N_ROWS] ; prints TV tables (all rows, top-10% / top-1% entropy rows) and the greedy bf16-vs-fp32 mismatch rate."""
import json
import sys

import torch

sys.path.insert(0, "/mnt/tt-data/ssinghal/wt/hy4_samp")
from models.demos.blackhole.hy4_preview.tt.hy4_sampler import emulate_keep_weights

PATH = "/mnt/tt-data/ssinghal/hy4-logs/program/lm-head/final_ref.pt"
COLS, SHARD = 8, 120832 // 8


def exact_p(l, T, k=0, p=1.0):
    s = l.double() / T
    s = s - s.max()
    e = s.exp()
    if k and k > 0:
        thr = torch.topk(s, min(k, s.numel())).values[-1]
        e = e * (s >= thr)
    if p < 1.0:
        srt, idx = torch.sort(e, descending=True)
        c = srt.cumsum(0)
        keep = (c - srt) < p * c[-1]  # nucleus: smallest prefix with mass >= p
        m = torch.zeros_like(e)
        m[idx[keep]] = 1
        e = e * m
    return e / e.sum()


def shared_p(l, T, k=0, p=1.0, bf16=True):
    kk = 32 if (k < 1 or k > 32) else k
    v = l.bfloat16() if bf16 else l.float()
    cand = torch.cat([torch.topk(v[c * SHARD : (c + 1) * SHARD].float(), 32).indices + c * SHARD for c in range(COLS)])
    cv = v[cand].float()
    top = torch.topk(cv, kk)
    ids, vals = cand[top.indices], top.values
    invT = torch.tensor(1.0 / T).bfloat16().float().item() if bf16 else 1.0 / T
    s = (vals - vals.max()) * invT
    e = s.double().exp()
    if p < 1.0:
        c = e.cumsum(0)
        e = e * ((c - e) < p * c[-1])
    out = torch.zeros(l.numel(), dtype=torch.float64)
    out[ids] = e / e.sum()
    return out


def thr_p(l, T, k=0, p=1.0):
    w = emulate_keep_weights(l, T, top_k=k, top_p=p, J=32, levels=4, levels_k=5).double()
    return w / w.sum()


def tv(a, b):
    return 0.5 * (a - b).abs().sum().item()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3385
    d = torch.load(PATH, map_location="cpu")
    L = d["sub_logits"][:n]
    print("rows", L.shape)
    # greedy: bf16-rounded logits argmax (lowest index among equal bf16 values, like a stable bf16 sampler) vs fp32 argmax
    g32 = L.argmax(-1)
    gb = L.bfloat16().float().argmax(-1)
    print(
        "greedy bf16-vs-fp32 mismatch: %d/%d = %.2f%%"
        % ((g32 != gb).sum(), len(g32), 100.0 * (g32 != gb).float().mean())
    )
    ent = []
    for i in range(len(L)):
        p = exact_p(L[i], 0.9)
        ent.append(-(p * (p + 1e-30).log()).sum().item())
    ent = torch.tensor(ent)
    order = ent.argsort(descending=True)
    sets = {
        "all": torch.arange(len(L)),
        "top10%ent": order[: max(1, len(L) // 10)],
        "top1%ent": order[: max(1, len(L) // 100)],
    }
    print("entropy (nats, T=0.9): median %.3f p90 %.3f max %.3f" % (ent.median(), ent.quantile(0.9), ent.max()))
    cases = [
        ("T0.9 k=off p=1 (generation_config default)", 0.9, 0, 1.0),
        ("T1.0 k=off p=1", 1.0, 0, 1.0),
        ("T0.9 k=20", 0.9, 20, 1.0),
        ("T0.9 k=32", 0.9, 32, 1.0),
        ("T0.9 k=50 p=0.9", 0.9, 50, 0.9),
        ("T0.9 k=off p=0.95", 0.9, 0, 0.95),
    ]
    sub = order[: max(1, len(L) // 10)]
    res = {}
    for name, T, k, p in cases:
        rows = {}
        for sname, idx in sets.items():
            if sname == "all":
                idx = torch.arange(0, len(L), max(1, len(L) // 200))  # 200 evenly spaced rows (thr is slow on CPU)
            tvs = {"shared": [], "shared32": [], "thr": []}
            for i in idx.tolist():
                ex = exact_p(L[i], T, k, p)
                tvs["shared"].append(tv(ex, shared_p(L[i], T, k, p)))
                tvs["shared32"].append(tv(ex, shared_p(L[i], T, k, p, bf16=False)))
                tvs["thr"].append(tv(ex, thr_p(L[i], T, k, p)))
            rows[sname] = {m: (round(sum(v) / len(v), 5), round(max(v), 4)) for m, v in tvs.items()}
        res[name] = rows
        print(name)
        for sname, r in rows.items():
            print("   %-10s" % sname, "  ".join("%s mean %.5f max %.4f" % (m, a, b) for m, (a, b) in r.items()))
    json.dump(res, open("/mnt/tt-data/ssinghal/hy4-logs/program/device-sampling/tv_cpu.json", "w"), indent=1)


if __name__ == "__main__":
    main()
