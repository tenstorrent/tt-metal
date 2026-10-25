"""L02 (mHC lever) microbench: traced us/call of the mHC ops and of the layer-level mHC chain (the sequence DSV41Layer.forward runs, attention / MoE removed).
T=4 (k=3 verify), T=6 (k=5 verify, odd_t -> 8), T=8, T=1 (plain decode). Env DSV41_MHC_LEVER_T = comma list (default 4,6,8,1)."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC, _odd_t, _pad_rows, _slice_rows

D = 5120


def traced_us(md, fn, n=40, reps=5):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        best = min(best, (time.perf_counter() - t0) / n * 1e6)
    ttnn.release_trace(md, tid)
    return best


def layer_chain(ma, mf, T, x, pre_in, a, m, sh, wa, wf):
    """exactly the mHC calls of DSV41Layer.forward (decode path)"""
    Tp = _odd_t(T)
    if Tp:
        P0 = lambda t: _pad_rows(t, T, Tp, 0)
        P2 = lambda t: _pad_rows(t, T, Tp, 2)
        S0 = lambda t: _slice_rows(t, T, 0)
        S2 = lambda t: _slice_rows(t, T, 2)
    else:
        P0 = P2 = S0 = S2 = lambda t: t
    xp, prep = P0(x), P0(pre_in)
    ap, apo, ac = ma.mixes(xp)
    h = S2(ma.collapse_norm(xp, prep, wa, 1e-20))
    x2, (fp, fpo, fc) = ma.expand_mixes(P2(a), xp, apo, ac, None, mf)
    h2, h_tok = mf.collapse_norm_rm(x2, ap, wf, 1e-20)
    h2, h_tok = S2(h2), S0(h_tok)
    x3 = S0(mf.expand(P2(m), x2, fpo, fc, P2(sh)))
    return ttnn.to_memory_config(x3, ttnn.DRAM_MEMORY_CONFIG), ttnn.to_memory_config(S0(fp), ttnn.DRAM_MEMORY_CONFIG)


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
def test_mhc_lever(mesh_device):
    md = mesh_device
    torch.manual_seed(1)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    mk = lambda: DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    ma, mf = mk(), mk()
    mi1 = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]), iters=1)
    wa, wf = up(torch.rand(1, 1, 1, D) + 0.5), up(torch.rand(1, 1, 1, D) + 0.5)
    for T in [int(v) for v in os.environ.get("DSV41_MHC_LEVER_T", "4,6,8,1").split(",")]:
        x = up(torch.randn(T, 1, 4, D))
        pre = up(torch.rand(T, 1, 1, 4))
        a = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
        m = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
        sh = up(torch.randn(1, 1, T, D))
        post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))
        y = up(torch.randn(T, 1, 1, D))
        res = {}
        res["mixes"] = traced_us(md, lambda: ma.mixes(x))
        if os.environ.get("DSV41_MHC_LEVER_IT1", "0") == "1":
            res["mixes_it1"] = traced_us(md, lambda: mi1.mixes(x))
        res["cn_rm"] = traced_us(md, lambda: ma.collapse_norm_rm(x, pre, wa, 1e-20))
        res["expand"] = traced_us(md, lambda: ma.expand(a, x, post, comb, None))
        res["chain"] = traced_us(md, lambda: layer_chain(ma, mf, T, x, pre, a, m, sh, wa, wf), n=20)
        print(
            "MHCLEVER T=%d " % T + "  ".join(f"{k}={v:.1f}" for k, v in res.items()) + "  (us/call, traced)", flush=True
        )
