"""mHC fusion bench: per-op chain time (traced, single-op repeated) + PCC of the whole mixes/collapse_norm/expand vs the old reference impl."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.mhc_ref_impl import DSV41MHC as Ref
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_opt import chain_ms, first, pcc
from models.demos.blackhole.deepseek_v41_flash.tt import mhc_mixes
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC as New

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
def test_fuse_bench(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    fn, base, scale = torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5])
    ref, new = Ref(md, fn, base, scale), New(md, fn, base, scale)
    tt_w = up(torch.rand(1, 1, 1, D) + 0.5)
    for kind in ("random", "realistic"):
        if kind == "random":
            x = torch.randn(T, 1, 4, D)
        else:
            x = torch.randn(T, 1, 1, D) + 0.3 * torch.randn(T, 1, 4, D)
            x[..., :4] *= 300.0
            x[..., 777] *= 800.0
        tx = up(x)
        pre_in = up(torch.softmax(torch.randn(T, 1, 1, 4), -1) * 2)
        rp, rpo, rc = ref.mixes(tx)
        npre, npo, nc = new.mixes(tx)
        y = torch.randn(1, 1, T, D).to(torch.bfloat16)
        ty = up(y, ttnn.bfloat16)
        y2 = up(torch.randn(1, 1, T, D))
        res = {
            "pre": pcc(first(npre), first(rp)),
            "post": pcc(first(npo), first(rpo)),
            "comb": pcc(first(nc), first(rc)),
            "maxdcomb": float((first(nc) - first(rc)).abs().max()),
            "maxdpre": float((first(npre) - first(rp)).abs().max()),
        }
        rn = ttnn.rms_norm(
            ttnn.typecast(ttnn.reshape(ref.collapse(tx, pre_in), [1, 1, T, D]), ttnn.bfloat16),
            epsilon=1e-20,
            weight=tt_w,
        )
        res["cn"] = pcc(first(new.collapse_norm(tx, pre_in, tt_w)), first(rn))
        res["exp"] = pcc(
            first(new.expand(ty, tx, npo, nc, y2)),
            first(ref.expand(up((y.float() + 0).reshape(T, 1, 1, D) + first(y2).reshape(T, 1, 1, D)), tx, rpo, rc)),
        )
        print(f"BENCH PCC[{kind}] T={T}", {k: round(v, 6) for k, v in res.items()}, flush=True)
    # timings
    plan = mhc_mixes.proj_plan(T, 5120, md)
    wt = new._wt(plan)
    part = mhc_mixes.mhc_proj(tx, wt, new.mix_col, plan)
    t = lambda name, f: print(f"BENCH TIME T={T} {name:22s} {chain_ms(md, f) * 1e3:7.1f} us", flush=True)
    t("mixes (proj+post)", lambda: new.mixes(tx))
    t("proj", lambda: mhc_mixes.mhc_proj(tx, wt, new.mix_col, plan))
    t(
        "post",
        lambda: mhc_mixes.mhc_post(part, new.consts9, plan, new._w.iters, new._w.eps, new.sq_eps, new.post_fidelity),
    )
    t("collapse_norm", lambda: new.collapse_norm(tx, pre_in, tt_w))
    t("collapse_norm_rm", lambda: new.collapse_norm_rm(tx, pre_in, tt_w))
    t("expand(bf16 y)", lambda: new.expand(ty, tx, npo, nc))
    t("expand(bf16 y, y2)", lambda: new.expand(ty, tx, npo, nc, y2))
