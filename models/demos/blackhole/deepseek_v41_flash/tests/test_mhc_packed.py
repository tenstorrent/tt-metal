# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed-layout mHC (tt/mhc_packed.py) vs the current T=32 padded kernels: PCC and traced time. Env PK_N (tokens, default 512)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_microbench import traced_ms
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_packed import PackedMHC

D = 5120
N = int(os.environ.get("PK_N", "512"))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 200_000_000}], indirect=True)
@torch.no_grad()
def test_mhc_packed(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t.contiguous(), device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    fn = torch.randn(24, 4 * D) * 0.02
    mhc = DSV41MHC(md, fn, torch.randn(24), torch.tensor([0.5, 0.5, 0.5]))
    pk = PackedMHC(mhc)
    nch = N // 32
    xh = torch.randn(N, 4, D)
    xs = [up(xh[c * 32 : (c + 1) * 32].reshape(32, 1, 4, D)) for c in range(nch)]
    yh = torch.randn(N, D)
    y2h = torch.randn(N, D)
    ys = [up(yh[c * 32 : (c + 1) * 32].reshape(1, 1, 32, D)) for c in range(nch)]
    y2s = [up(y2h[c * 32 : (c + 1) * 32].reshape(1, 1, 32, D)) for c in range(nch)]
    nw = up(torch.rand(1, 1, 1, D) + 0.5)
    xp = up(xh.reshape(1, 1, N, 4 * D))
    yp, y2p = up(yh.reshape(1, 1, N, D)), up(y2h.reshape(1, 1, N, D))
    th = lambda t: ttnn.to_torch(t).float()
    # reference = current kernels
    mix0 = [mhc.mixes(x) for x in xs]
    cur_pre = torch.cat([th(m[0]).reshape(32, 4) for m in mix0])
    cur_post = torch.cat([th(m[1]).reshape(32, 4) for m in mix0])
    cur_comb = torch.cat([th(m[2]).reshape(32, 16) for m in mix0])
    pre, post, comb = pk.mixes(xp)
    print(
        "PK pcc pre",
        pcc(th(pre).reshape(N, 4), cur_pre),
        "post",
        pcc(th(post).reshape(N, 4), cur_post),
        "comb",
        pcc(th(comb).reshape(N, 16), cur_comb),
        flush=True,
    )
    # collapse + norm with the SAME coefficients (current pre)
    pre_p = up(cur_pre.reshape(1, 1, N, 4))
    cn_cur = torch.cat([th(mhc.collapse_norm_rm(x, m[0], nw)[0]).reshape(32, D) for x, m in zip(xs, mix0)])
    h_ref = torch.einsum("ti,tid->td", cur_pre, xh)
    h_pk = th(pk.collapse(xp, pre_p)).reshape(N, D)
    print("PK pcc collapse(h) vs torch", pcc(h_pk, h_ref), "maxabs", float((h_pk - h_ref).abs().max()), flush=True)
    wref = th(nw).reshape(D)
    nref = (
        h_ref.bfloat16().float() * torch.rsqrt((h_ref.bfloat16().float() ** 2).mean(-1, keepdim=True) + 1e-6)
    ) * wref
    print("PK cur cn vs torch norm", pcc(cn_cur, nref), flush=True)
    cn_pk = th(pk.collapse_norm(xp, pre_p, nw, 1e-6)).reshape(N, D)
    print("PK pk cn vs torch norm", pcc(cn_pk, nref), flush=True)
    print("PK pcc collapse_norm", pcc(cn_pk, cn_cur), "maxabs", float((cn_pk - cn_cur).abs().max()), flush=True)
    # expand with the same coefficients
    post_p, comb_p = up(cur_post.reshape(1, 1, N, 4)), up(cur_comb.reshape(1, 1, N, 16))
    ex_cur = torch.cat(
        [th(mhc.expand(y, x, m[1], m[2], y2)).reshape(32, 4 * D) for y, y2, x, m in zip(ys, y2s, xs, mix0)]
    )
    ex_pk = th(pk.expand(yp, xp, post_p, comb_p, y2p)).reshape(N, 4 * D)
    print("PK pcc expand(+y2)", pcc(ex_pk, ex_cur), "maxabs", float((ex_pk - ex_cur).abs().max()), flush=True)
    ex_ref = torch.einsum("tij,tid->tjd", cur_comb.reshape(N, 4, 4), xh) + cur_post[:, :, None] * (yh + y2h)[:, None, :]
    print(
        "PK pcc expand vs torch",
        pcc(ex_pk, ex_ref.reshape(N, 4 * D)),
        "cur vs torch",
        pcc(ex_cur, ex_ref.reshape(N, 4 * D)),
        flush=True,
    )
    ex2 = th(pk.expand(yp, xp, post_p, comb_p)).reshape(N, 4 * D)
    ex_ref2 = torch.einsum("tij,tid->tjd", cur_comb.reshape(N, 4, 4), xh) + cur_post[:, :, None] * yh[:, None, :]
    print("PK pcc expand(no y2) vs torch", pcc(ex2, ex_ref2.reshape(N, 4 * D)), flush=True)
    # timing
    res = {}
    res["cur mixes x nch"] = traced_ms(md, lambda: [mhc.mixes(x) for x in xs])
    res["cur cn_rm x nch"] = traced_ms(md, lambda: [mhc.collapse_norm_rm(x, m[0], nw) for x, m in zip(xs, mix0)])
    res["cur expand x nch"] = traced_ms(
        md, lambda: [mhc.expand(y, x, m[1], m[2], y2) for y, y2, x, m in zip(ys, y2s, xs, mix0)]
    )
    res["pk mixes"] = traced_ms(md, lambda: pk.mixes(xp))
    res["pk proj kernel only"] = traced_ms(
        md,
        lambda: __import__("models.demos.blackhole.deepseek_v41_flash.tt.mhc_packed", fromlist=["x"]).pk_proj(
            xp, pk.wt, 24
        ),
    )
    ctp = pk.coefs_pre(pre_p)
    res["pk coefs_pre"] = traced_ms(md, lambda: pk.coefs_pre(pre_p))
    res["pk collapse (kernel+coef)"] = traced_ms(md, lambda: pk.collapse(xp, pre_p))
    res["pk collapse_norm"] = traced_ms(md, lambda: pk.collapse_norm(xp, pre_p, nw, 1e-6))
    res["pk expand y+y2"] = traced_ms(md, lambda: pk.expand(yp, xp, post_p, comb_p, y2p))
    res["pk expand y"] = traced_ms(md, lambda: pk.expand(yp, xp, post_p, comb_p))
    for k, v in res.items():
        print(f"MB {k:30s} {v:8.3f} ms  ({v * 1e3 / N:6.2f} us/token)", flush=True)
