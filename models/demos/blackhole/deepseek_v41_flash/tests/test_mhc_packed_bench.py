# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC for PREFILL: current T=32 padded-stream kernels (looped over N/32 chunks) vs packed [1,1,N,4C] composites incl. the
deepseek_prefill mhc_split_sinkhorn op. Single device, N tokens (env PK_N, default 512 = 4096 tok/row colsplit over 8 columns). Traced ms per call."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_microbench import traced_ms
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC

D = 5120
N = int(os.environ.get("PK_N", "512"))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 200_000_000}], indirect=True)
@torch.no_grad()
def test_mhc_packed_bench(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t.contiguous(), device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    fn = torch.randn(24, 4 * D) * 0.02
    base = torch.randn(24)
    scale = torch.tensor([0.5, 0.5, 0.5])
    mhc = DSV41MHC(md, fn, base, scale)
    w = mhc._w
    nch = N // 32
    xh = torch.randn(N, 4, D)
    xs = [up(xh[c * 32 : (c + 1) * 32].reshape(32, 1, 4, D)) for c in range(nch)]
    yh = torch.randn(N, D)
    ys = [up(yh[c * 32 : (c + 1) * 32].reshape(1, 1, 32, D)) for c in range(nch)]
    nw = up(torch.rand(1, 1, 1, D) + 0.5)
    mix0 = [mhc.mixes(x) for x in xs]
    for x, m in zip(xs, mix0):
        mhc.collapse_norm_rm(x, m[0], nw)
        mhc.expand(ys[0], x, m[1], m[2])
    res = {}

    def cur_mixes():
        return [mhc.mixes(x) for x in xs]

    def cur_cn():
        return [mhc.collapse_norm_rm(x, m[0], nw) for x, m in zip(xs, mix0)]

    def cur_exp():
        return [mhc.expand(y, x, m[1], m[2]) for y, x, m in zip(ys, xs, mix0)]

    res["cur mixes"] = traced_ms(md, cur_mixes)
    res["cur collapse_norm_rm"] = traced_ms(md, cur_cn)
    res["cur expand"] = traced_ms(md, cur_exp)
    # ---- packed composites
    xp = up(xh.reshape(1, 1, N, 4 * D))
    yp = up(yh.reshape(1, 1, N, D))
    mixes_p = lambda: w.project(xp)
    mx = mixes_p()
    res["pk project (matmul+ss)"] = traced_ms(md, mixes_p)
    sk = lambda: ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mx, w.consts, w.n, w.iters, w.eps)
    pre, post, comb = sk()
    res["pk split_sinkhorn"] = traced_ms(md, sk)
    r4 = lambda t: ttnn.reshape(t, [1, 1, N, t.shape[-1]])
    pre4, post4, comb4 = r4(pre), r4(post), r4(comb)
    from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _cols, _mix, _streams

    def collapse():
        h = _mix(_streams(xp, 4), _cols(pre4, 4))
        return ttnn.typecast(ttnn.rms_norm(h, epsilon=1e-6, weight=nw), ttnn.bfloat16)

    res["pk collapse+norm composite"] = traced_ms(md, collapse)
    res["pk expand composite"] = traced_ms(md, lambda: w.hc_post(yp, xp, post4, comb4))
    for k, v in res.items():
        print(f"MB {k:30s} {v:8.3f} ms  ({v * 1e3 / N:6.2f} us/token)", flush=True)
    cur = res["cur mixes"] + res["cur collapse_norm_rm"] + res["cur expand"]
    pk = (
        res["pk project (matmul+ss)"]
        + res["pk split_sinkhorn"]
        + res["pk collapse+norm composite"]
        + res["pk expand composite"]
    )
    print(f"MB TOTAL per sub-block N={N}: current {cur:.3f} ms, packed composite {pk:.3f} ms", flush=True)
    # correctness of packed pieces vs the current kernels
    cm = torch.cat([ttnn.to_torch(m[0]).float().reshape(32, 4) for m in mix0])
    print("MB pcc pre", pcc(ttnn.to_torch(pre4).float().reshape(N, 4), cm))
    ce = torch.cat([ttnn.to_torch(e).float().reshape(32, 4, D) for e in cur_exp()]).reshape(N, 4 * D)
    pe = ttnn.to_torch(w.hc_post(yp, xp, post4, comb4)).float().reshape(N, 4 * D)
    print("MB pcc expand", pcc(pe, ce))
