"""fused collapse+norm (mhc_collapse_norm) vs the two-op path: equality, PCC vs fp64 exact, chain timing. env MHC_T."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_opt import chain_ms, first, pcc
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC as New
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import mhc_collapse_norm

D = 5120
T = int(os.environ.get("MHC_T", "4"))


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
def test_cn_fused(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    fn, base, scale = torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5])
    new = New(md, fn, base, scale)
    wn = torch.rand(1, 1, 1, D) + 0.5
    tt_w = up(wn)
    w4 = new._norm_weight(tt_w, T)
    for kind in ("random", "realistic"):
        if kind == "random":
            x = torch.randn(T, 1, 4, D)
        else:
            x = torch.randn(T, 1, 1, D) + 0.3 * torch.randn(T, 1, 4, D)
            x[..., :4] *= 300.0
            x[..., 777] *= 800.0
        tx = up(x)
        pre_t = torch.softmax(torch.randn(T, 1, 1, 4), -1) * 2
        pre = up(pre_t)
        old = first(new.collapse_norm(tx, pre, tt_w))
        f = mhc_collapse_norm(tx, pre, w4, 1e-20)
        fr, frm = mhc_collapse_norm(tx, pre, w4, 1e-20, emit_rm=True)
        hx = (pre_t.double().reshape(T, 4, 1) * x.double().reshape(T, 4, D)).sum(1).to(torch.bfloat16).double()
        ex = hx * torch.rsqrt((hx * hx).mean(-1, keepdim=True) + 1e-20) * wn.double().reshape(1, D)
        print(
            f"CN T={T} {kind}: fused vs old pcc {pcc(first(f), old):.6f} maxdiff {float((first(f) - old).abs().max()):.4f}; fused vs exact {pcc(first(f), ex):.6f}; old vs exact {pcc(old, ex):.6f}; "
            f"rm pcc {pcc(first(frm).reshape(T, D), old.reshape(T, D)):.6f} rm==tile {float((first(frm).reshape(T, D) - first(fr).reshape(T, D)).abs().max()):.4f}",
            flush=True,
        )
    for _ in range(3):
        a = mhc_collapse_norm(tx, pre, w4, 1e-20)
        assert torch.equal(first(a), first(f))
    t = lambda name, fn_: print(f"CN TIME T={T} {name:20s} {chain_ms(md, fn_) * 1e3:7.1f} us", flush=True)
    t("old collapse_norm", lambda: new.collapse_norm(tx, pre, tt_w))
    t("old collapse_norm_rm", lambda: new.collapse_norm_rm(tx, pre, tt_w))
    t("fused", lambda: mhc_collapse_norm(tx, pre, w4, 1e-20))
    t("fused rm", lambda: mhc_collapse_norm(tx, pre, w4, 1e-20, emit_rm=True))
