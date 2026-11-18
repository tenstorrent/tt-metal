"""B49 unit (1x8 slot): fused prefill iHC (hy4_ihc_pf.Hy4IHCPF: pre -> (h, st), expand) vs the stock Hy4IHCGen.pre_post + collapse + rms_exact + expand on random streams (n=512, D=6144).
Metrics: pre/post max abs err, h (bf16) PCC / max abs / fraction of 1-ulp-or-more diffs, x4' max rel err; traced us of both; soak ITER repeated calls vs the first output (bit-equal).
Env: SEED, ITER (soak, default 1), AMP (stream scale), OUTLIER=1 (massive-activation channels like the bos sink)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4IHCGen, rms_exact
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_pf import Hy4IHCPF

N, D = 512, 6144


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def tr(md, fn, reps=3, n=5):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return round(min(ts), 1)


def snap(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def main():
    full, md = open_row()
    try:
        g = torch.Generator().manual_seed(int(os.environ.get("SEED", "1")))
        amp = float(os.environ.get("AMP", "2.0"))
        fn8 = torch.randn(8, 4 * D, generator=g) * 0.02
        base8 = torch.randn(8, generator=g) * 0.3
        scale2 = torch.tensor([0.7, 0.4])
        wn = (1.0 + 0.2 * torch.randn(D, generator=g)).bfloat16().float()
        eps_n, hc_eps = 1e-6, 1e-6
        ihc = Hy4IHCGen(md, fn8, base8, scale2, eps_n, hc_eps)
        pf = Hy4IHCPF(md, ihc.w, base8, scale2, wn, hc_eps, eps_n)
        w32 = upload(md, wn.reshape(1, 1, 1, D), ttnn.float32)
        xh = torch.randn(1, 4, N, D, generator=g) * amp
        if os.environ.get("OUTLIER") == "1":
            xh[:, :, 0, 100] = 3000.0
            xh[:, :, :, 5] += 50.0
        x = upload(md, xh, ttnn.float32)
        yh = torch.randn(1, 1, N, D, generator=g) * 0.5
        y = upload(md, yh, ttnn.bfloat16)
        if os.environ.get("EARLY") == "1":
            outs = []
            for k in range(4):
                hh, ss_ = pf.pre(x)
                ttnn.synchronize_device(md)
                outs.append((snap(hh), snap(ss_), snap(pf.sh["part"]), snap(pf.sh["accp"])))
            res(
                event="early",
                h_eq=[bool(torch.equal(outs[0][0], o[0])) for o in outs],
                st_eq=[bool(torch.equal(outs[0][1], o[1])) for o in outs],
                st_maxdiff=[float((outs[0][1] - o[1]).abs().max()) for o in outs],
                part_maxdiff=[float((outs[0][2] - o[2]).abs().max()) for o in outs],
                part_absmax=float(outs[0][2].abs().max()),
                accp_maxdiff=[float((outs[0][3] - o[3]).abs().max()) for o in outs],
            )
        # stock
        pre, post = ihc.pre_post(x)
        os.environ["HY4_B49_IHC"] = "1"
        raw = Hy4IHCGen.collapse(x, pre)
        h_ref = rms_exact(raw, w32, eps_n)
        x_ref = Hy4IHCGen.expand(ttnn.clone(x), y, ttnn.clone(post))
        pre_h, post_h, h_r, x_r = (
            snap(pre).reshape(4, N),
            snap(post).reshape(4, N),
            snap(h_ref).reshape(N, D),
            snap(x_ref).reshape(4, N, D),
        )
        # fused
        t0 = time.time()
        h, st = pf.pre(x)
        ttnn.synchronize_device(md)
        res(event="first_call_s", s=round(time.time() - t0, 1))
        if os.environ.get("DBG") == "1":
            P = snap(pf.sh["part"]).reshape(N // 32, 6, 2, 32, 32)  # [r, ch, {M,Q}, tok, col]
            m_dev = P[:, :, 0].sum(1)[:, :, :8].reshape(N, 8)  # [t, j]
            q_dev = P[:, :, 1].sum(1).sum(-1).reshape(N)
            m_ref = torch.einsum("std,jsd->tj", xh[0].double(), fn8.reshape(8, 4, D).double())
            q_ref = (xh[0].double() ** 2).sum((0, 2))
            m_st = snap(
                ttnn.sum(ttnn.matmul(x, ihc.w, compute_kernel_config=ihc.ckc, dtype=ttnn.float32), dim=1, keepdim=True)
            ).reshape(N, 8)
            q_st = snap(ttnn.sum(ttnn.sum(ttnn.multiply(x, x), dim=3, keepdim=True), dim=1, keepdim=True)).reshape(N)
            res(
                event="dbg_stock_stats",
                m_relerr=float((m_st - m_ref).abs().max() / m_ref.abs().max()),
                q_relerr=float(((q_st - q_ref).abs() / q_ref).max()),
            )
            res(
                event="dbg_stats",
                m_relerr=float((m_dev - m_ref).abs().max() / m_ref.abs().max()),
                q_relerr=float(((q_dev - q_ref).abs() / q_ref).max()),
                m0_dev=m_dev[0].tolist(),
                m0_ref=m_ref[0].tolist(),
            )
        h_f = snap(h).reshape(N, D)
        stt = snap(st).reshape(N // 32, 32, 32)[:, :, :8]  # [r, t, k]
        sk = stt.reshape(N, 8).T
        pre_f, post_f = sk[:4], sk[4:]
        xo = pf.expand(x, y, st)
        ttnn.synchronize_device(md)
        x_f = snap(xo).reshape(4, N, D)
        d_h = (h_f - h_r).abs()
        ulp = (h_f.bfloat16().view(torch.int16).int() - h_r.bfloat16().view(torch.int16).int()).abs()
        res(
            event="accuracy",
            pre_maxabs=float((pre_f - pre_h).abs().max()),
            post_maxabs=float((post_f - post_h).abs().max()),
            pre_range=[float(pre_h.min()), float(pre_h.max())],
            h_pcc=pcc(h_f, h_r),
            h_maxabs=float(d_h.max()),
            h_ref_absmax=float(h_r.abs().max()),
            h_frac_ulp_ge1=float((ulp >= 1).float().mean()),
            h_frac_ulp_ge2=float((ulp >= 2).float().mean()),
            x_maxrel=float(((x_f - x_r).abs() / (x_r.abs() + 1e-3)).max()),
            x_maxabs=float((x_f - x_r).abs().max()),
            x_pcc=pcc(x_f, x_r),
        )
        # fp64 truth for pre/post/h
        xd = xh[0].double()
        mm = torch.einsum("std,jsd->tj", xd, fn8.reshape(8, 4, D).double())
        rr = torch.rsqrt((xd**2).sum((0, 2)) / (4 * D) + eps_n)
        a64 = torch.sigmoid(
            mm * rr[:, None] * torch.tensor([scale2[0].item()] * 4 + [scale2[1].item()] * 4).double() + base8.double()
        )
        a64 = a64 * torch.tensor([1.0] * 4 + [2.0] * 4).double() + torch.tensor([hc_eps] * 4 + [0.0] * 4).double()
        pre64, post64 = a64[:, :4].T, a64[:, 4:].T
        raw64 = (pre64[:, :, None] * xd).sum(0)
        h64 = raw64 * torch.rsqrt((raw64**2).mean(-1, keepdim=True) + eps_n) * wn.double()
        x64 = xd + post64[:, :, None] * yh[0, 0].double()
        res(
            event="vs_fp64",
            stock_pre=float((pre_h - pre64).abs().max()),
            fused_pre=float((pre_f - pre64).abs().max()),
            stock_post=float((post_h - post64).abs().max()),
            fused_post=float((post_f - post64).abs().max()),
            stock_h_maxabs=float((h_r - h64).abs().max()),
            fused_h_maxabs=float((h_f - h64).abs().max()),
            stock_h_pcc=pcc(h_r, h64),
            fused_h_pcc=pcc(h_f, h64),
            stock_x_maxabs=float((x_r - x64).abs().max()),
            fused_x_maxabs=float((x_f - x64).abs().max()),
        )
        # soak
        iters = int(os.environ.get("ITER", "1"))
        bad = 0
        for i in range(iters - 1):
            h2, st2 = pf.pre(x)
            x2 = pf.expand(x, y, st2)
            ttnn.synchronize_device(md)
            eh, ex = torch.equal(snap(h2), snap(h)), torch.equal(snap(x2), snap(xo))
            if not (eh and ex):
                bad += 1
                if bad <= 3:
                    res(
                        event="soak_diff",
                        it=i,
                        h_equal=eh,
                        x_equal=ex,
                        h_maxdiff=float((snap(h2) - snap(h)).abs().max()),
                        x_maxdiff=float((snap(x2) - snap(xo)).abs().max()),
                        st_maxdiff=float((snap(st2) - snap(st)).abs().max()),
                        h_vs_ref=float((snap(h2).reshape(N, D) - h_r).abs().max()),
                    )
            ttnn.deallocate(h2)
            ttnn.deallocate(st2)
            ttnn.deallocate(x2)
        res(event="soak", iters=iters, mismatches=bad)
        if os.environ.get("TIME", "1") == "1":

            def stock():
                p_, q_ = ihc.pre_post(x)
                r_ = Hy4IHCGen.collapse(x, p_)
                return rms_exact(r_, w32, eps_n), ttnn.add(x, ttnn.multiply(q_, ttnn.typecast(y, ttnn.float32)))

            res(
                event="time_us",
                stock_pre_plus_expand=tr(md, stock, reps=1),
                fused_pre=tr(md, lambda: pf.pre(x), reps=1),
                fused_expand=tr(md, lambda: pf.expand(x, y, st), reps=1),
            )
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
