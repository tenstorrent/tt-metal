"""CPU numerics of the fused cross-column merge (tt/hy4_merge.py, mrg_compute.cpp): the kernel's formula with the kernel's roundings (bf16 ms / l / o inputs, w and rden packed to bf16) vs the float64 full softmax with the sink.
w_c = l_c exp(ms_c - sink); den = sum w_c + 1; out = (sum_c o_c w_c) / den, o_c normalised by the local sum, ms scaled max, sink in the scaled domain (valid while ms - sink < 88)."""
import torch

bf = lambda x: x.to(torch.bfloat16).to(torch.float64)


def partials(scores, v, ncol):
    """split the keys of one head row over ncol columns -> per column (o normalised by local sum, ms = max, l = sum exp(s - max)); empty column: ms = -inf, l = 0"""
    n = scores.numel()
    cols = torch.arange(n) % ncol
    o, ms, l = [], [], []
    for c in range(ncol):
        s = scores[cols == c]
        if s.numel() == 0:
            o.append(torch.zeros(v.shape[1], dtype=torch.float64)); ms.append(float("-inf")); l.append(0.0)
            continue
        m = s.max()
        p = torch.exp(s - m)
        o.append((p[:, None] * v[cols == c]).sum(0) / p.sum()); ms.append(float(m)); l.append(float(p.sum()))
    return torch.stack(o), torch.tensor(ms, dtype=torch.float64), torch.tensor(l, dtype=torch.float64)


def merge_kernel(o, ms, l, sink):
    o, ms, l = bf(o), bf(ms), bf(l)
    w = bf(l * torch.exp(torch.clamp(ms - sink, max=80.0)))  # packed to bf16 (CB_W)
    rden = bf(1.0 / (w.sum() + 1.0))  # bf16 CB_R
    return bf((o * w[:, None]).sum(0) * rden)


def ref(scores, v, sink):
    m = torch.maximum(scores.max(), torch.tensor(sink, dtype=torch.float64))
    p = torch.exp(scores - m)
    return (p[:, None] * v).sum(0) / (p.sum() + torch.exp(torch.tensor(sink, dtype=torch.float64) - m))


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def run(n, ncol, scale_logit, sink, seed=0, empty=False):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(n, generator=g, dtype=torch.float64) * scale_logit
    v = torch.randn(n, 64, generator=g, dtype=torch.float64)
    if empty:
        n = 5  # fewer keys than columns: columns >= 5 are empty
        scores, v = scores[:n], v[:n]
    o, ms, l = partials(scores, v, ncol)
    got = merge_kernel(o, ms, l, sink)
    return pcc(got, ref(scores, v, sink)), torch.isfinite(got).all().item()


def test_random_and_peaked():
    outs = []
    for seed in range(8):
        for sl in (1.0, 3.0, 8.0):
            p, ok = run(2048, 8, sl, sink=0.3, seed=seed)
            assert ok and p > 0.999, (seed, sl, p)
            outs.append(p)
    assert sum(outs) / len(outs) > 0.9995


def test_empty_columns():
    p, ok = run(5, 8, 2.0, sink=-1.0, empty=True)
    assert ok and p > 0.999


def test_hot_logits_below_overflow_margin():
    p, ok = run(2048, 8, 12.0, sink=-2.0, seed=3)  # max scaled logit ~ 40
    assert ok and p > 0.999
