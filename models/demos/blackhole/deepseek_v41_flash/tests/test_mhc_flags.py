"""mHC env-flag variants (DSV41_MHC_MIXES_V2 / _EXPAND_V2 / _CN_FUSED) vs the reference implementation: PCC + traced chain timing.
env MHC_T (default 4), MHC_CFGS (":"-separated lists of flags, e.g. "base;MIXES_V2;MIXES_V2,EXPAND_V2,CN_FUSED")."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.mhc_ref_impl import DSV41MHC as Ref
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_opt import chain_ms, first, pcc
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC as New

D = 5120
T = int(os.environ.get("MHC_T", "4"))
CFGS = os.environ.get("MHC_CFGS", "base:MIXES_V2:EXPAND_V2:CN_FUSED").split(":")


def setflags(cfg):
    for k in ("MIXES_V2", "EXPAND_V2", "CN_FUSED", "EP"):
        os.environ.pop("DSV41_MHC_" + k, None)
    if cfg != "base":
        for k in cfg.split(","):
            os.environ["DSV41_MHC_" + k] = "1"


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
def test_mhc_flags(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    fn, base, scale = torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5])
    ref, new = Ref(md, fn, base, scale), New(md, fn, base, scale)
    tt_w = up(torch.rand(1, 1, 1, D) + 0.5)
    inputs = {}
    for kind in ("random", "realistic"):
        if kind == "random":
            x = torch.randn(T, 1, 4, D)
        else:
            x = torch.randn(T, 1, 1, D) + 0.3 * torch.randn(T, 1, 4, D)
            x[..., :4] *= 300.0
            x[..., 777] *= 800.0
        inputs[kind] = (up(x), up(torch.softmax(torch.randn(T, 1, 1, 4), -1) * 2))
    y = torch.randn(1, 1, T, D).to(torch.bfloat16)
    ty, y2 = up(y, ttnn.bfloat16), up(torch.randn(1, 1, T, D))
    for cfg in CFGS:
        setflags(cfg)
        for kind, (tx, pre_in) in inputs.items():
            rp, rpo, rc = ref.mixes(tx)
            npre, npo, nc = new.mixes(tx)
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
            h, htok = new.collapse_norm_rm(tx, pre_in, tt_w)
            res["cn_rm"] = pcc(first(htok).reshape(T, D), first(rn).reshape(T, D))
            res["exp"] = pcc(
                first(new.expand(ty, tx, npo, nc, y2)),
                first(ref.expand(up((y.float() + 0).reshape(T, 1, 1, D) + first(y2).reshape(T, 1, 1, D)), tx, rpo, rc)),
            )
            xr = ref.expand(up((y.float() + 0).reshape(T, 1, 1, D) + first(y2).reshape(T, 1, 1, D)), tx, rpo, rc)
            for yy, tag in ((y2, "y2"), (None, "noy2")):
                xr_ = xr if yy is not None else ref.expand(up(y.float().reshape(T, 1, 1, D)), tx, rpo, rc)
                xn, (ep, epo, ec) = new.expand_mixes(ty, tx, npo, nc, yy, new)
                rpx, rpox, rcx = ref.mixes(xr_)
                xnt = first(xn)
                res["ep_" + tag + "_x"] = pcc(xnt, first(xr_))
                res["ep_" + tag + "_mm"] = pcc(
                    first(new.collapse(xn, pre_in)), first(new.collapse(xr_, pre_in))
                )  # padded rows must be zero
                res["ep_" + tag + "_mix"] = min(
                    pcc(first(ep), first(rpx)), pcc(first(epo), first(rpox)), pcc(first(ec), first(rcx))
                )
            print(f"FLAGS PCC cfg={cfg} [{kind}] T={T}", {k: round(v, 6) for k, v in res.items()}, flush=True)
        print(f"FLAGS cfg={cfg} ep_used={getattr(new, 'ep_used', 0)}", flush=True)
        tx, pre_in = inputs["random"]
        npre, npo, nc = new.mixes(tx)
        t = lambda name, f: print(f"FLAGS TIME cfg={cfg} T={T} {name:18s} {chain_ms(md, f) * 1e3:7.1f} us", flush=True)
        t("mixes", lambda: new.mixes(tx))
        t("collapse_norm", lambda: new.collapse_norm(tx, pre_in, tt_w))
        t("collapse_norm_rm", lambda: new.collapse_norm_rm(tx, pre_in, tt_w))
        t("expand(+y2)", lambda: new.expand(ty, tx, npo, nc, y2))
        t("expand_mixes(+y2)", lambda: new.expand_mixes(ty, tx, npo, nc, y2, new))
        t("expand+mixes sep", lambda: new.mixes(new.expand(ty, tx, npo, nc, y2)))
    setflags("base")
