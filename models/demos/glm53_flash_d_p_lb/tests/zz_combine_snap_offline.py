"""Offline: can the combine's fp32 accumulation be made order-independent by snapping every term to a fixed absolute
grid (per layer, per hidden channel) before the add? GLM-5.3-Flash, real weights, golden ffn_norm + routing.

term[t, k, c] = bf16(w_tk) * bf16(y_tk[c]) in fp32 (what the accumulator's DEST holds before the packer adds it).
Calibrate B_c = max_t sum_k |term| (bounds every partial sum, any order) on s4096 (tokens 0-4095, both chunks),
test on s56320 chunk 10 (tokens 51200-56319).
Grid: E_c = ceil(log2 B_c) + h, fp32 q = 2^(E_c - 24) (every partial sum exact), int32 q = 2^(E_c - 31).
"""
import math
import sys
import time

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.golden import Golden, hf_path, load_spec
from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig
from models.demos.glm53_flash_d_p.reference.weights import PackedExpert, WeightLoader

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
spec = load_spec()
path = hf_path(spec)
cfg = GlmConfig.from_json(f"{path}/config.json")
wl = WeightLoader(path)
cal = Golden.for_rung(spec, "s4096")
tst = Golden.for_rung(spec, "s56320")
LIM, K = cfg.swiglu_limit, cfg.num_experts_per_tok
HMAX = LIM * float(F.silu(torch.tensor(LIM)))  # |silu(clamp(g, max=L)) * clamp(u, -L, L)| <= L * silu(L)


def terms(gl, W):
    """[T, K, H] fp32 terms, slots in expert-id order per token."""
    x, r = gl["ffn_norm"].float(), gl["router"].float()
    T, H = x.shape
    tok, exp = (r != 0).nonzero(as_tuple=True)  # row-major: per token, expert-id ascending
    slot = torch.arange(len(tok)) - torch.searchsorted(tok, tok)  # rank within the token
    assert int(slot.max()) < K
    out = torch.zeros(T, K, H, dtype=torch.float32)
    for e in torch.unique(exp).tolist():
        m = exp == e
        tt, ss = tok[m], slot[m]
        wg, wu, wd = W[e]
        xe = x[tt]
        h = F.silu(F.linear(xe, wg).clamp(max=LIM)) * F.linear(xe, wu).clamp(min=-LIM, max=LIM)
        y = F.linear(h, wd).bfloat16().float()
        out[tt, ss] = y * r[tt, e, None].bfloat16().float()
    return out


def bf(t):
    return t.float().bfloat16().double()


def report(i, W, wd_l1):
    t0 = time.time()
    B = torch.zeros(cfg.hidden_size, dtype=torch.float64)
    arg_tok = None
    for c in cal.dumped_chunks:
        tm = terms(cal.layer(c, i), W)
        s = tm.double().abs().sum(1)
        B = torch.maximum(B, s.amax(0))
        if arg_tok is None or s.max() >= B.max():
            arg_tok = (c, int(s.amax(1).argmax()))
    hard = 2.5 * HMAX * wd_l1  # sum w = 2.5, |h| <= HMAX, |y_c| <= ||Wd_e[c, :]||_1 * HMAX
    loose = torch.log2(hard / B)

    tm = terms(tst.layer(10, i), W)
    td = tm.double()
    exact = td.sum(1)
    sabs = td.abs().sum(1)
    E0 = torch.ceil(torch.log2(B))
    lines = [
        f"[snap] L{i}: B_layer 2^{math.log2(B.max()):.2f} (cal max at chunk/token {arg_tok}), "
        f"B_c spread log2 p50 {E0.median():.0f} min {E0.min():.0f} max {E0.max():.0f}; "
        f"hard bound loose by log2 p50 {loose.median():.1f} min {loose.min():.1f}"
    ]
    over = sabs / B
    lines.append(
        f"   test sum|term| / B_c: max 2^{math.log2(over.max()):.2f}; frac > 1x {(over > 1).double().mean():.2e}, "
        f"> 2x {(over > 2).double().mean():.2e}, > 4x {(over > 4).double().mean():.2e}"
    )

    # today: fp32 sum, two orders
    fwd = torch.zeros_like(tm[:, 0])
    rev = torch.zeros_like(tm[:, 0])
    for k in range(K):
        fwd += tm[:, k]
        rev += tm[:, K - 1 - k]
    ref_err = (bf(exact) - exact).norm()  # bf16 output rounding of the exact sum
    en = exact.norm()
    lines.append(
        f"   fp32 sum: fwd vs rev bf16 outputs differ in {(bf(fwd) != bf(rev)).double().mean():.2e} of elements; "
        f"fp32 err / bf16 rounding {((fwd.double() - exact).norm() / ref_err):.3f}"
    )

    def snap(qlog2, sat):
        q = torch.pow(2.0, qlog2)
        s = (td / q).round() * q  # round half even, exact in fp64
        out = s.sum(1)
        if sat is not None:
            out = out.clamp(-sat, sat)
        return out

    for name, qlog2, sat in [
        ("fp32 grid c h=0", E0 - 24, 2.0**E0),
        ("fp32 grid c h=1", E0 + 1 - 24, 2.0 ** (E0 + 1)),
        ("fp32 grid c h=2", E0 + 2 - 24, 2.0 ** (E0 + 2)),
        ("fp32 grid layer h=1", (E0.max() + 1 - 24).expand_as(E0), None),
        ("int32 c h=1", E0 + 1 - 31, None),
        ("int32 c h=2", E0 + 2 - 31, None),
    ]:
        out = snap(qlog2, sat)
        pre = (out - exact).norm() / ref_err  # error before the bf16 cast, in units of bf16 rounding
        diff = (bf(out) != bf(exact)).double().mean()
        rel = (bf(out) - exact).norm() / en
        lines.append(
            f"   {name:20s}: pre-cast err / bf16 rounding {pre:.3f}; bf16 out != bf16(exact) {diff:.2e}; "
            f"rel L2 {rel:.3e} (bf16(exact) {ref_err / en:.3e}, fp32 fwd {(bf(fwd) - exact).norm() / en:.3e})"
        )
    print("\n".join(lines) + f"\n   ({time.time() - t0:.0f}s)", flush=True)


for i in [int(v) for v in sys.argv[1].split(",")]:
    t = time.time()
    W = [PackedExpert(wl, i, e).weights(torch.float32) for e in range(cfg.n_routed_experts)]
    wd_l1 = torch.stack([w[2].double().abs().sum(1) for w in W]).amax(0)
    print(f"[snap] L{i}: weights {time.time() - t:.0f}s", flush=True)
    report(i, W, wd_l1)
    del W
