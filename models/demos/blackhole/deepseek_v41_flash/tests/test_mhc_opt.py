# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""New DSV41MHC (tt/mhc.py) vs the previous implementation (tests/mhc_ref_impl.py): PCC of mixes/collapse/collapse_norm/
expand on random and realistic (massive-activation) streams, plus chain-style traced timing."""
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.mhc_ref_impl import DSV41MHC as Ref
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC as New

D, T = 5120, 4


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def traced_ms(mesh, fn, n=20):
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


def chain_ms(mesh, f):
    def rp(k):
        def g():
            for _ in range(k):
                f()

        return g

    return (traced_ms(mesh, rp(3)) - traced_ms(mesh, rp(1))) / 2


def first(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


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
def test_mhc_opt(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    fn = torch.randn(24, 4 * D) * 0.02
    base = torch.randn(24) * 0.3
    scale = torch.tensor([0.4, 0.3, 0.5])
    ref, new = Ref(md, fn, base, scale), New(md, fn, base, scale)
    wn32 = torch.rand(1, 1, 1, D) + 0.5
    tt_w = up(wn32)

    def make_x(kind):
        if kind == "random":
            return torch.randn(T, 1, 4, D)
        # realistic: highly correlated streams, a few massive-activation channels
        base_ = torch.randn(T, 1, 1, D)
        x = base_ + 0.3 * torch.randn(T, 1, 4, D)
        x[..., :4] *= 300.0
        x[..., 777] *= 800.0
        return x

    ok = True
    for kind in ("random", "realistic"):
        x = make_x(kind)
        tt_x = up(x)
        pre_in = torch.softmax(torch.randn(T, 1, 1, 4), -1) * 2
        tt_pre_in = up(pre_in)
        y = torch.randn(T, 1, 1, D)
        tt_y = up(y)
        rp, rpo, rc = ref.mixes(tt_x)
        npre, npo, nc = new.mixes(tt_x)
        res = {
            "pre": pcc(first(npre), first(rp)),
            "post": pcc(first(npo), first(rpo)),
            "comb": pcc(first(nc), first(rc)),
        }
        rcol = first(ref.collapse(tt_x, tt_pre_in))
        res["collapse"] = pcc(first(new.collapse(tt_x, tt_pre_in)), rcol)
        # reference norm exactly as layer._norm (fused rms_norm on the bf16 collapse)
        rn = ttnn.rms_norm(
            ttnn.typecast(ttnn.reshape(ref.collapse(tt_x, tt_pre_in), [1, 1, T, D]), ttnn.bfloat16),
            epsilon=1e-20,
            weight=tt_w,
        )
        nn_ = new.collapse_norm(tt_x, tt_pre_in, tt_w)
        res["collapse_norm"] = pcc(first(nn_), first(rn))
        # torch exact for collapse_norm (fp64) to see which of ref/new is closer
        hx = (pre_in.double().reshape(T, 4, 1) * x.double().reshape(T, 4, D)).sum(1)
        hx = hx.to(torch.bfloat16).double()
        ex = hx * torch.rsqrt((hx * hx).mean(-1, keepdim=True) + 1e-20) * wn32.double().reshape(1, D)
        res["cn_new_vs_exact"] = pcc(first(nn_), ex)
        res["cn_ref_vs_exact"] = pcc(first(rn), ex)
        res["expand"] = pcc(first(new.expand(tt_y, tt_x, rpo, rc)), first(ref.expand(tt_y, tt_x, rpo, rc)))
        res["expand_own"] = pcc(first(new.expand(tt_y, tt_x, npo, nc)), first(ref.expand(tt_y, tt_x, rpo, rc)))
        res["max|dcomb|"] = float((first(nc) - first(rc)).abs().max())
        print(f"PCC[{kind}]", {k: round(v, 6) for k, v in res.items()}, flush=True)

    tt_x = up(make_x("realistic"))
    tt_pre_in = up(torch.rand(T, 1, 1, 4))
    tt_y = up(torch.randn(T, 1, 1, D))
    pre, post, comb = new.mixes(tt_x)
    print(f"TIME new mixes          {chain_ms(md, lambda: new.mixes(tt_x)):.3f} ms")
    print(f"TIME ref mixes          {chain_ms(md, lambda: ref.mixes(tt_x)):.3f} ms")
    print(f"TIME new collapse_norm  {chain_ms(md, lambda: new.collapse_norm(tt_x, tt_pre_in, tt_w)):.3f} ms")
    ref_cn = lambda: ttnn.rms_norm(
        ttnn.typecast(ttnn.reshape(ref.collapse(tt_x, tt_pre_in), [1, 1, T, D]), ttnn.bfloat16),
        epsilon=1e-20,
        weight=tt_w,
    )
    print(f"TIME ref collapse+norm  {chain_ms(md, ref_cn):.3f} ms")
    print(f"TIME new expand         {chain_ms(md, lambda: new.expand(tt_y, tt_x, post, comb)):.3f} ms")
    print(f"TIME ref expand         {chain_ms(md, lambda: ref.expand(tt_y, tt_x, post, comb)):.3f} ms")
