"""Sampler correctness on real rows: device draw vs CPU inverse-CDF of softmax(logits/T) in vocabulary order (exact same u), greedy rows, per-user seed determinism / slot independence, top-p / top-k paths, and the empirical distribution (TV distance)."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])
T = max(TS)
head = Hy4DeviceHead(md, mc, ccl, W, head_dtype="bf16", out_dtype="fp32", fidelity="HiFi4")
smp = Hy4DeviceSampler(md, mc, ccl, T)
smp.alloc_params()


def cpu_draw(l, temp, tk, tp, u):
    p = torch.softmax(l.double() / temp, 0)
    if tk > 0 or tp < 1:
        s, i = p.sort(descending=True)
        if tk > 0:
            s, i = s[:tk], i[:tk]
        s = s / s.sum()
        if tp < 1:
            s = s * ((s.cumsum(0) - s) < tp)
        q = torch.zeros_like(p)
        q[i] = s
        p = q
    cum = p.cumsum(0)
    return int(
        torch.searchsorted(cum, torch.tensor(float(u), dtype=torch.float64) * cum[-1], right=True).clamp(
            max=l.numel() - 1
        )
    )


def run_step(lg, temp, tk, tp, gens, greedy=0.0):
    invT, k, p, u, g = sampling_rows(temp, tk, tp, gens, T)
    if greedy:
        g = torch.ones(T)
    smp.set_params(invT, k, p, u, g)
    tok = smp.run(lg, smp.params, smp.fast)
    out = ttnn.to_torch(tok, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=1))  # [T, 8]? identical on all devices
    return out.reshape(T, -1)[:, 0].long(), u


rows = list(range(0, 16 * 40, 5))[: T * 8]
cfgs = [
    ("plain T0.9", 0.9, 0, 1.0),
    ("topp0.95", 0.9, 0, 0.95),
    ("topk20", 0.9, 20, 1.0),
    ("topk50+topp0.9", 0.9, 50, 0.9),
    ("T1.0 plain", 1.0, 0, 1.0),
]
agg = {c[0]: [0, 0] for c in cfgs}
agg["greedy"] = [0, 0]
cdf_dev = (
    []
)  # |u * total - CDF(device token)| in probability mass: how far the device draw is from the exact inverse CDF (plain rows)
for i in range(0, len(rows), T):
    ch = X[rows[i : i + T]]
    lg = head.forward(streams_tt(ch))
    full = head.gather_logits(lg)
    for nm, temp, tk, tp in cfgs:
        gens = [torch.Generator().manual_seed(100 + 7 * j + i) for j in range(T)]
        tok, u = run_step(lg, temp, tk, tp, gens)
        for j in range(T):
            ref = cpu_draw(full[j], temp, tk, tp, u[j])
            if tk == 0 and tp == 1.0:
                pj = torch.softmax(full[j].double() / temp, 0)
                cj = pj.cumsum(0)
                d = int(tok[j])
                lo, hi = (cj[d - 1] if d > 0 else 0.0), cj[d]
                uu = float(u[j]) * float(cj[-1])
                cdf_dev.append(max(float(lo) - uu, uu - float(hi), 0.0))
            agg[nm][0] += int(tok[j]) == ref
            agg[nm][1] += 1
            if tk == 0 and tp == 1.0 and int(tok[j]) != ref and agg[nm][1] - agg[nm][0] <= 3:
                res(
                    event="mismatch",
                    cfg=nm,
                    u=float(u[j]),
                    dev=int(tok[j]),
                    ref=ref,
                    lp_dev=float(full[j][int(tok[j])]),
                    lp_ref=float(full[j][ref]),
                )
    tok, _ = run_step(lg, 0.9, 0, 1.0, torch.Generator().manual_seed(1), greedy=1.0)
    agg["greedy"][0] += int((tok == full.argmax(-1)).sum())
    agg["greedy"][1] += T
res(
    event="cdf_gap",
    n=len(cdf_dev),
    max=max(cdf_dev),
    mean=sum(cdf_dev) / len(cdf_dev),
    frac_exact=sum(1 for c in cdf_dev if c == 0.0) / len(cdf_dev),
)
for k, (a, b) in agg.items():
    res(event="smp_match", cfg=k, match=a, n=b, frac=a / b)

# per-user seeds: the same seed in different slots / steps gives the same token (the draw does not depend on slot or batch composition)
ch = X[:T]
lg = head.forward(streams_tt(ch))
full = head.gather_logits(lg)
seeds = [11 + j for j in range(T)]
tk1, _ = run_step(lg, 0.9, 0, 1.0, [torch.Generator().manual_seed(s) for s in seeds])
perm = torch.randperm(T, generator=torch.Generator().manual_seed(5)).tolist()
# permute the rows: row p gets the logits and the seed of user perm[p]
lg2 = head.forward(streams_tt(ch[perm]))
tk2, _ = run_step(lg2, 0.9, 0, 1.0, [torch.Generator().manual_seed(seeds[p]) for p in perm])
same = all(int(tk2[p]) == int(tk1[perm[p]]) for p in range(T))
res(event="seed_slot_independent", same=same)
tk3, _ = run_step(lg, 0.9, 0, 1.0, [torch.Generator().manual_seed(s) for s in seeds])
res(event="seed_repeatable", same=bool((tk3 == tk1).all()))

# empirical distribution (TV distance) on 2 real rows, N draws each (flat-ish and peaked)
N = int(os.environ.get("NDRAW", "4000"))
for r in (3, 17):
    full_r = head.gather_logits(head.forward(streams_tt(X[r : r + 1].repeat(T, 1, 1))))[0]
    p = torch.softmax(full_r.double() / 0.9, 0)
    cnt = torch.zeros(VOCAB, dtype=torch.float64)
    g = torch.Generator().manual_seed(99)
    lg = head.forward(streams_tt(X[r : r + 1].repeat(T, 1, 1)))
    for _ in range(N // T):
        tok, _ = run_step(lg, 0.9, 0, 1.0, g)
        for t in tok.tolist():
            cnt[t] += 1
    emp = cnt / cnt.sum()
    res(
        event="tv",
        row=r,
        N=int(cnt.sum()),
        tv=float(0.5 * (emp - p).abs().sum()),
        expected_tv_noise=float(0.5 * (p * (1 - p) / cnt.sum()).sqrt().sum() * 0.8),
        top1_p=float(p.max()),
        top1_emp=float(emp[p.argmax()]),
    )
ttnn.close_mesh_device(md)
