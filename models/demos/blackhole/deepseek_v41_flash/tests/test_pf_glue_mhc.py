# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""T19: packed-mHC glue fusions (tt/pf_glue.py) vs the unfused packed ops on one chip: mixes (DSV41_PF_GLUE_MIX), collapse_norm (DSV41_PF_GLUE_NORM),
expand with bf16 y operands (DSV41_PF_GLUE_TC). Prints exact-equality fraction, PCC, max abs diff and traced us per call. Env G_N = tokens (128 = 4k window, 512 = 16k)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_microbench import traced_ms
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_packed import PackedMHC

D = 5120
N = int(os.environ.get("G_N", "128"))
ENVS = ("DSV41_PF_GLUE_MIX", "DSV41_PF_GLUE_NORM", "DSV41_PF_GLUE_TC", "DSV41_PF_GLUE_FUSE")


def _set(**kw):
    for k in ENVS:
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ[k] = v


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 200_000_000}], indirect=True)
@torch.no_grad()
def test_pf_glue_mhc(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t.contiguous(), device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    th = lambda t: ttnn.to_torch(t).float()
    fn = torch.randn(24, 4 * D) * 0.02
    mhc = DSV41MHC(md, fn, torch.randn(24), torch.tensor([0.5, 0.5, 0.5]))
    pk = PackedMHC(mhc)
    xh = torch.randn(N, 4, D) * 2
    xp = up(xh.reshape(1, 1, N, 4 * D))
    nw = up(torch.rand(1, 1, 1, D) + 0.5)
    yh = torch.randn(N, D)
    y2h = torch.randn(N, D)
    y_bf = ttnn.from_torch(yh.reshape(1, 1, N, D), device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y2p = up(y2h.reshape(1, 1, N, D))
    res = {}
    if (
        os.environ.get("G_SWEEP") == "1"
    ):  # device-profiler run: kernel durations of pk_proj / pk_mix for several K splits, in this order
        from models.demos.blackhole.deepseek_v41_flash.tt import pf_glue
        from models.demos.blackhole.deepseek_v41_flash.tt.mhc_packed import pk_proj

        for S_ in (64, 32, 16, 8):
            for _ in range(2):
                pa = pk_proj(xp, pk.wt, 24, S=S_)
                mx = pf_glue.pk_mix(pa, 24, 1.0 / (4 * D), 1e-6)
            ttnn.synchronize_device(md)
            print(
                "SWEEP S",
                S_,
                "mixes equal to S=64:",
                bool((th(mx) == th(pf_glue.pk_mix(pk_proj(xp, pk.wt, 24, S=64), 24, 1.0 / (4 * D), 1e-6))).all()),
                flush=True,
            )
        ct = pk.coefs_pre(m_pre_dummy := up(torch.rand(N, 4).reshape(1, 1, N, 4)))
        for mc in (130, 64, 32):
            for K_ in (8, 16, 32):
                h_, part_, nrc_ = pf_glue.pk_collapse_sq(xp, ct, max_cores=mc)
                o_ = pf_glue.pk_norm(h_, part_, nw, 1e-6, k_split=K_)
            ttnn.synchronize_device(md)
            print("SWEEP collapse_sq max_cores", mc, "nrc", nrc_, flush=True)
        return

    if (
        os.environ.get("G_SWEEP2") == "1"
    ):  # profiler run: pk_comb kernel time vs number of cores (max_cores), expand and collapse, order: for mc in MCS: expand(y,y2), collapse
        from models.demos.blackhole.deepseek_v41_flash.tt.mhc_packed import pk_comb

        cc_ = up(torch.rand(N, 16).reshape(1, 1, N, 16))
        cp_ = up(torch.rand(N, 4).reshape(1, 1, N, 4))
        cce, cpe = (
            pk.coefs_expand(cp_ if False else up(torch.rand(N, 4).reshape(1, 1, N, 4)), cc_) if False else (None, None)
        )
        ce = ttnn.matmul(cc_, pk.e16, compute_kernel_config=pk.ckc)
        cpv = ttnn.matmul(cp_, pk.e4, compute_kernel_config=pk.ckc)
        cpre = pk.coefs_pre(cp_)
        for mc in (130, 96, 64, 48, 32, 24):
            for _ in range(2):
                o1 = pk_comb(xp, ce, cpv, y=y2p, y2=y2p, nout=4, max_cores=mc)
                o2 = pk_comb(xp, cpre, None, nout=1, out_dtype=ttnn.bfloat16, max_cores=mc)
            ttnn.synchronize_device(md)
            print("SWEEP2 max_cores", mc, flush=True)
        return

    def cmp(name, a, b):
        a, b = th(a), th(b)
        eq = float((a == b).float().mean())
        print(
            f"CMP {name}: equal {eq:.6f} pcc {pcc(a.flatten(), b.flatten()):.7f} maxabs {float((a - b).abs().max()):.3e} (ref absmax {float(b.abs().max()):.3e})",
            flush=True,
        )

    # mixes
    _set()
    m0 = pk.mixes(xp)
    _set(DSV41_PF_GLUE_MIX="1")
    m1 = pk.mixes(xp)
    for nme, a, b in zip(("pre", "post", "comb"), m1, m0):
        cmp("mixes." + nme, a, b)
    # collapse_norm
    pre = m0[0]
    _set()
    c0 = pk.collapse_norm(xp, pre, nw, 1e-6)
    _set(DSV41_PF_GLUE_NORM="1")
    c1 = pk.collapse_norm(xp, pre, nw, 1e-6)
    cmp("collapse_norm", c1, c0)
    # which arithmetic does ttnn.rms_norm use?  torch candidates on the bf16 collapse output
    ct_ = pk.coefs_pre(pre)
    from models.demos.blackhole.deepseek_v41_flash.tt.mhc_packed import pk_comb

    hh = th(pk_comb(xp, ct_, None, nout=1, out_dtype=ttnn.bfloat16)).reshape(N, D).bfloat16().float()
    ww = th(nw).reshape(-1)[:D]
    r0, r1_ = th(c0).reshape(N, D), th(c1).reshape(N, D)
    torch.save({"h": hh, "w": ww, "unfused": r0, "fused": r1_}, "/out/rms_dump.pt")
    ss32 = (hh * hh).sum(-1, keepdim=True)
    ss64 = (hh.double() ** 2).sum(-1, keepdim=True)
    for nm_, rsv in (
        ("fp32 sum, rsqrt(mean+eps)", torch.rsqrt(ss32 / D + 1e-6)),
        ("fp64 sum", torch.rsqrt((ss64 / D + 1e-6)).float()),
        ("fp32 sum*(1/D)", torch.rsqrt(ss32 * (1.0 / D) + 1e-6)),
    ):
        for ord_, o_ in (
            ("(h*rs)*w", ((hh * rsv) * ww).bfloat16().float()),
            ("h*(rs*w)", (hh * (rsv * ww)).bfloat16().float()),
        ):
            print(
                f"RMSAB {nm_:28s} {ord_}: unfused mismatches {int((o_ != r0).sum())}  fused mismatches {int((o_ != r1_).sum())} of {r0.numel()}",
                flush=True,
            )
    # expand with bf16 y (TC): reference = typecast first
    post, comb = m0[1], m0[2]
    _set()
    y32 = ttnn.typecast(y_bf, ttnn.float32)
    e0 = pk.expand(y32, xp, post, comb, y2p)
    _set(DSV41_PF_GLUE_TC="1")
    e1 = pk.expand(y_bf, xp, post, comb, y2p)
    cmp("expand(bf16 y)", e1, e0)
    if os.environ.get("G_NOTIME") == "1":
        return
    # timing: per call, traced
    for nme, env, fn_ in (
        ("mixes", "DSV41_PF_GLUE_MIX", lambda: pk.mixes(xp)),
        ("collapse_norm", "DSV41_PF_GLUE_NORM", lambda: pk.collapse_norm(xp, pre, nw, 1e-6)),
        (
            "expand+typecast(y)",
            "DSV41_PF_GLUE_TC",
            lambda: pk.expand(ttnn.typecast(y_bf, ttnn.float32), xp, post, comb, y2p),
        ),
    ):
        _set()
        a = traced_ms(md, fn_)
        _set(**{env: "1"})
        if env == "DSV41_PF_GLUE_TC":
            fn_ = lambda: pk.expand(y_bf, xp, post, comb, y2p)
        b = traced_ms(md, fn_)
        res[nme] = (a * 1e3, b * 1e3)
    # kernel-only timings: pk_proj for several K splits, pk_mix on partials of that many tiles
    from models.demos.blackhole.deepseek_v41_flash.tt import pf_glue
    from models.demos.blackhole.deepseek_v41_flash.tt.mhc_packed import pk_proj

    R = N // 32
    for S_ in (64, 32, 16):
        pa = pk_proj(xp, pk.wt, 24, S=S_)
        res[f"pk_proj S={S_}"] = (traced_ms(md, lambda: pk_proj(xp, pk.wt, 24, S=S_)) * 1e3, 0.0)
        res[f"pk_mix S={S_} (alone)"] = (0.0, traced_ms(md, lambda: pf_glue.pk_mix(pa, 24, 1.0 / (4 * D), 1e-6)) * 1e3)
    for k, (a, b) in res.items():
        print(f"MB N={N} {k:22s} unfused {a:8.1f} us  fused {b:8.1f} us  ({a - b:+.1f})", flush=True)
