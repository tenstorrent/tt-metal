# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced device-time micro-benchmarks of the mHC pieces (4x8 mesh, 4 tokens per device) to decide what to rewrite.
Prints MB lines: name = ms per replay (a trace of ONE call, 50 replays)."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC

D, T = 5120, 4


def traced_ms(mesh, fn, n=50):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t) / n
    ttnn.release_trace(mesh, tid)
    return dt * 1e3


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_mhc_microbench(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    fn = torch.randn(24, 4 * D) * 0.02
    mhc = DSV41MHC(md, fn, torch.randn(24), torch.tensor([0.5, 0.5, 0.5]))
    w = mhc._w
    x = up(torch.randn(1, 1, T, 4 * D))  # packed streams [1,1,T,4D]
    pre_in = up(torch.rand(1, 1, T, 4))
    pre, post, comb = mhc.mixes(x)
    y = up(torch.randn(1, 1, T, D))
    ckc = w.ckc
    res = {}
    # ---- current implementation, piece by piece
    res["mixes: matmul x@fn_T (fp32 HiFi4)"] = traced_ms(md, lambda: ttnn.matmul(x, w.fn_T, compute_kernel_config=ckc))
    res["mixes: sumsq (x*x, sum)"] = traced_ms(md, lambda: ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True))
    mix = ttnn.matmul(x, w.fn_T, compute_kernel_config=ckc)
    res["mixes: mhc_split_sinkhorn kernel alone"] = traced_ms(
        md, lambda: ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mix, w.consts, 4, w.iters, w.eps)
    )
    res["mixes: whole (current)"] = traced_ms(md, lambda: mhc.mixes(x))
    res["collapse (current, 12 ops)"] = traced_ms(md, lambda: mhc.collapse(x, pre_in))
    res["expand (current, ~45 ops)"] = traced_ms(md, lambda: mhc.expand(y, x, post, comb))
    # ---- candidate building blocks
    xb = up(torch.randn(T, 1, 4, D))  # tokens as batch, streams in tile rows
    cmat = up(torch.randn(T, 1, 4, 4))
    res["cand expand: batched matmul comb^T[T,4,4]@res[T,4,D]"] = traced_ms(
        md, lambda: ttnn.matmul(cmat, xb, compute_kernel_config=ckc)
    )
    pre_b = up(torch.randn(T, 1, 1, 4))
    res["cand collapse: batched matmul pre[T,1,4]@res[T,4,D]"] = traced_ms(
        md, lambda: ttnn.matmul(pre_b, xb, compute_kernel_config=ckc)
    )
    post_b = up(torch.randn(T, 1, 4, 1))
    yb = up(torch.randn(T, 1, 1, D))
    res["cand expand: post[T,4,1]*y[T,1,D] broadcast multiply"] = traced_ms(md, lambda: ttnn.multiply(post_b, yb))
    res["cand: add [T,4,D]+[T,4,D]"] = traced_ms(md, lambda: ttnn.add(xb, xb))
    fncat = up(torch.randn(1, 1, D, 96) * 0.02)
    res["cand mixes: matmul xb[T,4,D]@fn_cat[D,96]"] = traced_ms(
        md, lambda: ttnn.matmul(xb, fncat, compute_kernel_config=ckc)
    )
    xbf16 = ttnn.typecast(xb, ttnn.bfloat16)
    wn = up(torch.rand(1, 1, 1, D), ttnn.bfloat16)
    res["norm: ttnn.rms_norm bf16 [T,5120]"] = traced_ms(
        md, lambda: ttnn.rms_norm(ttnn.slice(xbf16, [0, 0, 0, 0], [T, 1, 1, D]), weight=wn, epsilon=1e-20)
    )
    for k, v in res.items():
        print(f"MB {k:62s} {v:7.3f} ms")
