# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: where the all-gather MoE path's output gain comes from (tt/experts_ag.py), on one MoE layer's golden.

Runs the block step by step (replicated input: no gather) and reads back chip 0's flat expert output y and its plan,
then for a few of chip 0's experts compares y with CPU references on the same bfp4 weights (dequantized fp8 ->
ttnn's host bfp4 quantizer): fp32 math on bf16 x (ref), and the same with x and / or h rounded to bfp8 on the host
(ttnn's host bfp8 quantizer). Prints the scale coefficient <y, ref> / <ref, ref> and rel L2 per variant, then the
coefficient of the reduced output (moe_ag_local_reduce + send-back + reduce over the columns) vs a host sum of the
device y (isolates the bf16 reductions). GLM_DIAG_LAYER (default 4), GLM_DIAG_EXPERTS (default 4)."""

import os

import torch

from models.demos.common.bringup.testing.component import _step
from models.demos.common.bringup.testing.harness import component_golden, device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
LAYER = int(os.environ.get("GLM_DIAG_LAYER", "4"))
N_EXP = int(os.environ.get("GLM_DIAG_EXPERTS", "4"))


def _coef(g, w):
    g, w = g.double().reshape(-1), w.double().reshape(-1)
    return float((g * w).sum() / (w * w).sum()), float((g - w).norm() / w.norm())


@mesh_parametrize
def test_flat_diag(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.reference.weights import PackedExpert
    from models.demos.glm53_flash_d_p.tt.common import replicate
    from models.demos.glm53_flash_d_p.tt.experts_ag import build_experts_ag

    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, "experts")
    gl = g.layer(c, LAYER)
    x, r = (gl[i].float() for i in st.inputs)
    s, H = x.shape[-2], x.shape[-1]
    hooks.apply_device_settings(S)
    loader, cfg = hooks._loader_cfg(S)
    mod = build_experts_ag(mesh_device, loader, cfg, LAYER, max(hooks._chunks(S)), weights_dtype=hooks.experts_dtype(S))

    xb = x.reshape(1, 1, s, H).to(torch.bfloat16)
    xd = replicate(mesh_device, xb)
    rd = replicate(mesh_device, r.reshape(1, 1, s, -1).to(torch.bfloat16))
    idx, wts = mod.topk_from_dense(rd)
    blk = mod._block(s // 2)
    gx, gi, gw, _ = blk.gather(xd, idx, wts, replicated=True)
    blk.plan(gi, blk.lmap)
    y = mod.flat(
        ttnn.reshape(gx, (blk.T, H)),
        blk.counts,
        blk.regions,
        token_index=blk.token_index,
        y_row_major=True,
        down_fp32=mod.down_fp32,
    )
    dev0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    y0 = dev0(y).float().reshape(-1, H)
    counts, regions = dev0(blk.counts).reshape(-1).long(), dev0(blk.regions).reshape(-1).long()
    tidx = dev0(blk.token_index).reshape(-1).long()

    q4 = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa
    q8 = lambda t: ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa
    act = lambda gg, uu: torch.nn.functional.silu(gg.clamp(max=10.0)) * uu.clamp(-10.0, 10.0)  # noqa: E731
    gids0 = mod.gids[0]
    order = sorted(gids0, key=lambda e: -int(counts[e]))
    picks = order[:2] + order[len(order) // 2 : len(order) // 2 + N_EXP - 2]
    pairs = {}  # name -> (got list, want list)

    def add(name, got, want):
        pairs.setdefault(name, ([], []))
        pairs[name][0].append(got)
        pairs[name][1].append(want)

    for e in picks:
        n, r0 = int(counts[e]), int(regions[e])
        n32 = -(-n // 32) * 32
        xe = xb.reshape(s, H).float()[tidx[r0 : r0 + n]]
        xe_pad = torch.cat([xe, torch.zeros(n32 - n, H)])
        gw_, uw_, dw_ = PackedExpert(loader, LAYER, e).weights(torch.float32)
        ex = [w.T.contiguous() for w in (gw_, uw_, dw_)]  # exact fp32 (Wg^T, Wu^T, Wd^T)
        A = [q4(w) for w in ex]  # bfp4(fp32): what unified holds
        B = [q4(w.bfloat16().float()) for w in ex]  # bfp4(bf16(fp32)): what the flat op holds (test_weight_bits.py)
        for nm, w, a, b in zip(("gate", "up", "down"), ex, A, B):
            dif = a != b
            nd = int(dif.sum())
            up = int((b.abs() > a.abs())[dif].sum())
            print(
                f"[diag] expert {e} {nm}: A!=B {nd} ({nd / a.numel():.5f}), |B|>|A| on {up}; "
                f"<A,W>/<W,W> {float((a * w).sum() / (w * w).sum()):.5f} <B,W>/<W,W> {float((b * w).sum() / (w * w).sum()):.5f} "
                f"sum|A|/sum|W| {float(a.abs().sum() / w.abs().sum()):.5f} sum|B|/sum|W| {float(b.abs().sum() / w.abs().sum()):.5f}",
                flush=True,
            )
        ye = y0[r0 : r0 + n]
        f = lambda W, xx=xe_pad, hq=None: (  # noqa: E731
            (hq(act(xx @ W[0], xx @ W[1])) if hq else act(xx @ W[0], xx @ W[1])) @ W[2]
        )[:n]
        rT, rA, rB = f(ex), f(A), f(B)
        add("y vs refB (its own weights)", ye, rB)
        add("y vs refB, bfp8 x and h", ye, f(B, q8(xe_pad), q8))
        add("y vs refA", ye, rA)
        add("y vs exact", ye, rT)
        add("refA vs exact (bfp4(fp32) weights)", rA, rT)
        add("refB vs exact (bfp4(bf16) weights)", rB, rT)
        add("refB vs refA", rB, rA)
        cf, rl = _coef(ye, rB)
        print(f"[diag] L{LAYER} chip0 expert {e} tokens {n}: y vs refB coef {cf:.5f} rel {rl:.5f}", flush=True)
    for k, (gs, ws) in pairs.items():
        cf, rl = _coef(torch.cat(gs), torch.cat(ws))
        print(f"[diag] all picked experts, {k}: coef {cf:.5f} rel {rl:.5f}", flush=True)

    # the reductions: local reduce + send-back + column sum vs a host sum of every chip's device y
    col = blk.reduce(y, gw)
    rs = ttnn.reduce_scatter(col, dim=2, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    half = ttnn.all_gather(rs, dim=2, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.all_gather(half, dim=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = dev0(out).float().reshape(s, H)
    w_host = dev0(gw).float().reshape(s, -1)
    i_host = dev0(gi).reshape(s, -1).long()
    want = torch.zeros(s, H, dtype=torch.float64)
    for d, (yt, ct, rg, ti) in enumerate(
        zip(
            ttnn.get_device_tensors(y),
            ttnn.get_device_tensors(blk.counts),
            ttnn.get_device_tensors(blk.regions),
            ttnn.get_device_tensors(blk.token_index),
        )
    ):
        yd = ttnn.to_torch(yt).double().reshape(-1, H)
        cd, rgd, tid = (ttnn.to_torch(t).reshape(-1).long() for t in (ct, rg, ti))
        for e in mod.gids[d]:
            n, r0 = int(cd[e]), int(rgd[e])
            if not n:
                continue
            toks = tid[r0 : r0 + n]
            k = (i_host[toks] == e).double().argmax(-1)
            wk = w_host[toks, k].double()
            want.index_add_(0, toks, yd[r0 : r0 + n] * wk[:, None])
    cf, rl = _coef(got, want)
    print(f"[diag] reduced output vs host fp64 sum of device y: coef {cf:.5f} rel {rl:.5f}", flush=True)
