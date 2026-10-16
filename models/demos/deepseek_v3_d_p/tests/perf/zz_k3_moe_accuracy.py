# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Offline accuracy analysis of Kimi K3 MoE blocks on the device (host only): reads dumps of the device blocks run
on one set of weights and one input (scratch_k3_moe_layer.py K3_DUMP=, the token-split block; zz_scratch_k3_ttmoe_
compare.py K3_DUMP_TTMOE=, the existing TtMoe) and grades each against
  * the upstream block (reference/kimi_k3/modeling_kimi_moe.KimiSparseMoeBlock, its experts pointed at the weight
    pool), in fp32 (the truth) and in bf16 (the HF model as run in bf16);
  * a host model with the device's weight formats (routed experts bfp4, dense bf8, router bf16: rounded exactly as
    ttnn does, 32 x 32 tiles of the [in, out] matrices), in fp32 math -- what a perfect device would compute;
and splits the error: weight quantization, routing (picks that differ, and how close they are to ties), device
compute per stage (each stage on the device's own input to it: local; and the host chain: accumulated).
Usage: python zz_k3_moe_accuracy.py --dump TOKENSPLIT.pt [--ttmoe TTMOE.pt] [--json OUT]"""

import argparse
import copy
import json

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import K3Dims, K3MoEWeights, random_k3_weights


def q(t, dt, bf16_first=False):
    """t rounded to a ttnn dtype on the host (tiles of the matrix as given). bf16_first: through bf16, as the flat
    expert's upload does (scratch_k3_moe_layer.py K3_WCHECK: its device weights are bfp4(bf16(W)) exactly; straight
    fp32 -> bfp4 differs in 0.75% of the elements, one bfp4 step). The block's dense weights and TtMoe's experts go
    straight (the matched reference gives the lower error)."""
    if dt == "bf16":
        return t.to(torch.bfloat16).float()
    d = {"bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}[dt]
    src = t.contiguous().float()
    if bf16_first:
        src = src.to(torch.bfloat16).float()
    return ttnn.to_torch(ttnn.from_torch(src, dtype=d, layout=ttnn.TILE_LAYOUT)).float()


def quantized(w: K3MoEWeights, wdtype="bf4", experts_bf16_first=True) -> K3MoEWeights:
    """The device's weights (dense [in, out] bf8, router bf16, experts [in, out] bfp4 or wdtype)."""
    return K3MoEWeights(
        gate=q(w.gate, "bf16"),
        bias=q(w.bias, "bf16"),
        lat_down=q(w.lat_down.T, "bf8").T,
        lat_up=q(w.lat_up.T, "bf8").T,
        lat_norm=q(w.lat_norm, "bf16"),
        sh_gate=q(w.sh_gate.T, "bf8").T,
        sh_up=q(w.sh_up.T, "bf8").T,
        sh_down=q(w.sh_down.T, "bf8").T,
        pool=[tuple(q(m.T, wdtype, experts_bf16_first).T for m in e) for e in w.pool],
    )


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp4(w, block=32):
    """w [out, in] rounded to MXFP4 (OCP MX: e2m1 elements, one power-of-two scale per 32 along the input dim),
    the format the K3 checkpoint ships its routed experts in."""
    o, i = w.shape
    b = w.reshape(o, i // block, block)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - 2)  # (e8m0: the block max lands in [4, 8) -> clamp to 6)
    v = (b / scale).abs().clamp(max=6.0)
    idx = (v.unsqueeze(-1) - _E2M1).abs().argmin(-1)
    return (_E2M1[idx] * b.sign() * scale).reshape(o, i)


def situ(g, u, d):
    return d.beta * torch.tanh(g / d.beta) * torch.sigmoid(g) * (d.linear_beta * torch.tanh(u / d.linear_beta))


def route(x, w, d):
    scores = torch.sigmoid(x @ w.gate.T)
    _, ids = torch.topk(scores + w.bias, d.K, dim=-1)
    wts = scores.gather(1, ids)
    return ids, wts / (wts.sum(-1, keepdim=True) + 1e-20) * d.route_scale


ROUTED_ACT = "situ"  # (the dump's routed activation: situ, or silu for the activation probe)


def routed_act(g, u, d):
    return torch.nn.functional.silu(g) * u if ROUTED_ACT == "silu" else situ(g, u, d)


def routed_sum(lat, w, d, ids, wts):
    acc = torch.zeros(lat.shape[0], d.latent)
    P = len(w.pool)
    for p in range(P):
        tok, kk = ((ids % P) == p).nonzero(as_tuple=True)
        if tok.numel():
            w1, w3, w2 = w.pool[p]
            h = lat[tok]
            acc.index_add_(0, tok, (routed_act(h @ w1.T, h @ w3.T, d) @ w2.T) * wts[tok, kk].float().unsqueeze(1))
    return acc


def norm_up(r, w, d):
    return (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + d.eps) * w.lat_norm) @ w.lat_up.T


def shared(x, w, d):
    return situ(x @ w.sh_gate.T, x @ w.sh_up.T, d) @ w.sh_down.T


def block(x, w, d, ids=None, wts=None):
    """The whole block in fp32 (the host model); every stage."""
    s = {}
    if ids is None:
        ids, wts = route(x, w, d)
    s["lat"] = x @ w.lat_down.T
    s["routed"] = routed_sum(s["lat"], w, d, ids, wts)
    s["yr"] = norm_up(s["routed"], w, d)
    s["ys"] = shared(x, w, d)
    s["y"] = s["yr"] + s["ys"]
    return s, ids, wts


def upstream(x, w, d, dtype):
    """The upstream KimiSparseMoeBlock on these weights (its E experts pointed at the P pool modules), in dtype."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiMoEGate, KimiSparseMoeBlock
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_hf_config

    P = len(w.pool)
    cfg = copy.deepcopy(kimi_k3_hf_config())
    cfg.num_experts = cfg.n_routed_experts = P
    cfg.rms_norm_eps = d.eps
    moe = KimiSparseMoeBlock(cfg)
    cg = copy.copy(cfg)
    cg.num_experts = cg.n_routed_experts = d.E
    moe.gate = KimiMoEGate(cg)
    with torch.no_grad():
        moe.gate.weight.copy_(w.gate)
        moe.gate.e_score_correction_bias.copy_(w.bias)
        for p, (w1, w3, w2) in enumerate(w.pool):
            moe.experts[p].w1.weight.copy_(w1)
            moe.experts[p].w3.weight.copy_(w3)
            moe.experts[p].w2.weight.copy_(w2)
        moe.experts = torch.nn.ModuleList([moe.experts[g % P] for g in range(d.E)])
        moe.routed_expert_down_proj.weight.copy_(w.lat_down)
        moe.routed_expert_up_proj.weight.copy_(w.lat_up)
        moe.routed_expert_norm.weight.copy_(w.lat_norm)
        moe.shared_experts.gate_proj.weight.copy_(w.sh_gate)
        moe.shared_experts.up_proj.weight.copy_(w.sh_up)
        moe.shared_experts.down_proj.weight.copy_(w.sh_down)
        moe = moe.eval().to(dtype)
        return moe(x.to(dtype).unsqueeze(0)).squeeze(0).float()


def stats(a, b):
    """a (test) vs b (reference): PCC, relative Frobenius error, per-token relative error percentiles, worst
    per-token cosine."""
    a, b = a.float(), b.float()
    pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    rel = ((a - b).norm() / b.norm()).item()
    tok = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    qs = torch.quantile(tok, torch.tensor([0.5, 0.99])).tolist()
    return dict(pcc=pcc, rel=rel, tok_p50=qs[0], tok_p99=qs[1], tok_max=tok.max().item(), cos_min=cos.min().item())


def fmt(st):
    return (f"PCC {st['pcc']:.6f}  rel {st['rel']:.2e}  per-token p50 {st['tok_p50']:.2e} p99 {st['tok_p99']:.2e} "
            f"max {st['tok_max']:.2e}  cos min {st['cos_min']:.5f}")


def routing_report(name, gi, gw, idsq, wtsq, x, wq, d, out, report):
    T = gi.shape[0]
    S_ = lambda ids: [set(r.tolist()) for r in ids]  # noqa: E731
    dev_s, q_s = S_(gi), S_(idsq)
    miss = torch.tensor([len(a_ - b_) for a_, b_ in zip(dev_s, q_s)])
    biased = torch.sigmoid(x @ wq.gate.T) + wq.bias
    kth = torch.topk(biased, d.K + 1, dim=-1).values
    gap = (kth[:, d.K - 1] - kth[:, d.K]).abs()
    mis_tok = (miss > 0).nonzero().flatten()
    print(f"  [{name}] picks differing from the host (bf16 router weight, fp32 math): {miss.sum().item()} of {T * d.K} "
          f"({100 * miss.sum().item() / (T * d.K):.2f}%), tokens affected {len(mis_tok)} of {T}, max per token {miss.max().item()}")
    if len(mis_tok):
        mass = []
        for t in mis_tok.tolist():
            only = dev_s[t] - q_s[t]
            mass.append(sum(gw[t, j].item() for j in range(d.K) if gi[t, j].item() in only))
        mass = torch.tensor(mass)
        print(f"  [{name}] 16th-17th biased-score gap: all tokens median {gap.median().item():.2e}, tokens with a "
              f"differing pick median {gap[mis_tok].median().item():.2e} max {gap[mis_tok].max().item():.2e}; gate weight "
              f"on the swapped experts median {mass.median().item():.4f} max {mass.max().item():.4f}")
    out[f"{name}/routing"] = dict(diff_picks=int(miss.sum()), tokens=int(len(mis_tok)))
    if gw is not None:
        pairs = []
        for t in range(T):
            hq = {int(i): float(v) for i, v in zip(idsq[t], wtsq[t])}
            pairs += [(gw[t, j].item(), hq[int(gi[t, j])]) for j in range(d.K) if int(gi[t, j]) in hq]
        pr = torch.tensor(pairs)
        report(f"[{name}] gate weights, common picks", stats(pr[:, :1], pr[:, 1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ttmoe")
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="skip the MXFP4 section")
    a = ap.parse_args()
    blocks = {"token-split": torch.load(a.dump)}
    if a.ttmoe:
        blocks["TtMoe"] = torch.load(a.ttmoe)
    D0 = blocks["token-split"]
    d = K3Dims(E=D0["E"])
    w = random_k3_weights(d, pool_size=D0["pool"])
    global ROUTED_ACT
    ROUTED_ACT = D0.get("act", "situ")
    print(f"routed activation: {ROUTED_ACT}")
    if ROUTED_ACT != "situ":
        a.ttmoe = None  # (the upstream / TtMoe references are SiTU: only the host model and per-stage checks apply)
    wq = quantized(w, D0.get("wdtype", "bf4"))
    print(f"routed expert weights on the device: {D0.get('wdtype', 'bf4')}")
    x = D0["x"].float()
    for nm, D in blocks.items():
        assert torch.equal(D["x"].float(), x), f"{nm}: a different input"
    out = {}

    def report(key, st):
        out[key] = st
        print(f"  {key:56s} {fmt(st)}")

    print("0. references:")
    ref, ids0, _ = block(x, w, d)
    refq, idsq, wtsq = block(x, wq, d)
    up32 = upstream(x, w, d, torch.float32)
    up16 = upstream(x, w, d, torch.bfloat16)
    report("host model vs upstream block (fp32)", stats(ref["y"], up32))
    report("upstream bf16 vs upstream fp32", stats(up16, up32))
    report("bf16 rounding of the fp32 output (floor)", stats(up32.to(torch.bfloat16), up32))
    report("quantized-weight model vs upstream fp32", stats(refq["y"], up32))
    for k in ("yr", "ys"):
        report(f"[{k}] quantized-weight model vs fp32 model", stats(refq[k], ref[k]))

    summary = {}
    for nm, D in blocks.items():
        print(f"\n=== {nm} ===")
        bf16_first = nm == "token-split"
        wq = quantized(w, D.get("wdtype", "bf4"), bf16_first)
        refq, idsq, wtsq = block(x, wq, d)
        print(f"  (reference experts rounded {'fp32 -> bf16 -> bfp' if bf16_first else 'fp32 -> bfp'}, as this block uploads them)")
        gi = D["gi"].long()
        gw = D["gw"].float() if "gw" in D else None
        y = D["y"].float()
        print("1. end to end (y):")
        st_up = stats(y, up32)
        report(f"[{nm}] device vs upstream fp32", st_up)
        report(f"[{nm}] device vs upstream bf16", stats(y, up16))
        report(f"[{nm}] device vs quantized-weight model (host routing)", stats(y, refq["y"]))
        if gw is not None:
            refq_dev, _, _ = block(x, wq, d, gi, gw)
            report(f"[{nm}] device vs quantized-weight model (device routing)", stats(y, refq_dev["y"]))
        print("2. routing:")
        if "logits" in D:
            report(f"[{nm}] router logits vs fp32 on the bf16 weight", stats(D["logits"], x @ wq.gate.T))
        routing_report(nm, gi, gw, idsq, wtsq, x, wq, d, out, report)
        print("3. per stage, local (host stage on the device's input to it) | accumulated (host chain, device routing):")
        report(f"[{nm}] latent down", stats(D["lat"], x @ wq.lat_down.T))
        if gw is not None:
            r_loc = routed_sum(D["lat"].float(), wq, d, gi, gw)
            report(f"[{nm}] routed experts + combine (local)", stats(D["routed"], r_loc))
            report(f"[{nm}] routed experts + combine (accumulated)", stats(D["routed"], refq_dev["routed"]))
            lat8 = q(D["lat"].float(), "bf8")
            report(f"[{nm}]   of which: bfp8 rounding of the tokens alone (host)", stats(routed_sum(lat8, wq, d, gi, gw), r_loc))
            report(f"[{nm}]   routed vs host on the bfp8 tokens", stats(D["routed"], routed_sum(lat8, wq, d, gi, gw)))
            report(f"[{nm}] latent norm + up (local)", stats(D["yr"], norm_up(D["routed"].float(), wq, d)))
            report(f"[{nm}] latent norm + up (accumulated)", stats(D["yr"], refq_dev["yr"]))
        report(f"[{nm}] shared expert", stats(D["ys"], refq["ys"]))
        report(f"[{nm}] sum (local)", stats(y, D["yr"].float() + D["ys"].float()))
        summary[nm] = st_up
    # --- the real checkpoint's routed experts are MXFP4: with MXFP4 weights as the truth, what the device's bfp4
    # costs (bfp4 shares an exponent per 16 along a tile row: the [in, out] matrix's output dim; MXFP4 per 32 along
    # the input dim), and what bfp4 on the transposed tiles would (exponent along the input dim, as MXFP4)
    if a.quick:
        if a.json:
            with open(a.json, "w") as f:
                json.dump(out, f, indent=1)
        return
    print("\n5. routed experts with MXFP4 weights as the truth (host, host routing):")
    lat = x @ w.lat_down.T
    ids, wts = route(x, w, d)
    w_mx = copy.copy(w)
    w_mx.pool = [tuple(mxfp4(m) for m in e) for e in w.pool]
    r_mx = routed_sum(lat, w_mx, d, ids, wts)
    for lbl, pool in (
        ("bfp4 of the MXFP4 weights (device layout, exponent along out)", [tuple(q(m.T, "bf4").T for m in e) for e in w_mx.pool]),
        ("bfp4 of the MXFP4 weights, exponent along in (transposed tiles)", [tuple(q(m, "bf4") for m in e) for e in w_mx.pool]),
        ("bfp8 of the MXFP4 weights (device layout)", [tuple(q(m.T, "bf8").T for m in e) for e in w_mx.pool]),
    ):
        wv = copy.copy(w)
        wv.pool = pool
        report(f"routed: {lbl}", stats(routed_sum(lat, wv, d, ids, wts), r_mx))
        wa, wb = torch.cat([m.flatten() for e in pool for m in e]), torch.cat([m.flatten() for e in w_mx.pool for m in e])
        print(f"  weights: {lbl:62s} rel {((wa - wb).norm() / wb.norm()).item():.2e}")
    report("routed: MXFP4 vs fp32 weights (the checkpoint's own loss)", stats(r_mx, routed_sum(lat, w, d, ids, wts)))
    report("routed: bfp4 vs fp32 weights (this test's random weights)", stats(routed_sum(lat, wq, d, ids, wts), routed_sum(lat, w, d, ids, wts)))
    if len(blocks) > 1:
        print("\n=== the blocks side by side, vs the upstream block in fp32 ===")
        for nm, st in summary.items():
            print(f"  {nm:12s} {fmt(st)}")
        names = list(blocks)
        report(f"{names[0]} vs {names[1]} (device to device)", stats(blocks[names[0]]["y"], blocks[names[1]]["y"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
