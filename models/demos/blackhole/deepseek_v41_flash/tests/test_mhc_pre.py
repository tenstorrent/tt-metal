"""L02 gate (a2): the one-program mixes + collapse + norm (DSV41MHC.mixes_collapse) is BIT-IDENTICAL to mixes() + collapse_norm[_rm]() for T = 1..8, and 9..16 with DSV41_MHC_PRE16=1."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC

D = 5120


def host(t):
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
def test_mhc_pre(mesh_device):
    os.environ["DSV41_MHC_PRE16"] = os.environ["DSV41_MHC_PRE24"] = "1"
    md = mesh_device
    torch.manual_seed(5)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    bad = tot = 0
    for trial in range(3):
        for T in range(1, 25):
            x_h = torch.randn(T, 1, 1, D) + 0.3 * torch.randn(T, 1, 4, D)
            x_h[..., :4] *= 300.0
            pre_h = torch.rand(T, 1, 1, 4)
            x, pre_in = up(x_h), up(pre_h)

            def refs(xh, ph):  # the plain per-row programs (unfused mixes + collapse_norm[_rm])
                xx, pp = up(xh), up(ph)
                n = xh.shape[0]
                rm_ = [host(o).reshape(n, -1) for o in m.mixes(xx)]
                rh_ = host(m.collapse_norm(xx, pp, w, 1e-20))[0, 0, :n]
                rh2, rtok = m.collapse_norm_rm(xx, pp, w, 1e-20)
                return rm_, rh_, host(rh2)[0, 0, :n], host(rtok).reshape(n, -1)

            if T <= 8:
                ref_mix, ref_h, ref_h2, ref_tok = refs(x_h, pre_h)
            else:  # T = 9..16: the reference of every row is the 4-token program of plain decode (what a row costs at T=4); groups of 4
                parts = [refs(x_h[lo : lo + 4], pre_h[lo : lo + 4]) for lo in range(0, T, 4)]
                ref_mix = [torch.cat([q[0][i] for q in parts], 0) for i in range(3)]
                ref_h, ref_h2, ref_tok = (torch.cat([q[k] for q in parts], 0) for k in (1, 2, 3))
            for emit_rm in (False, True):
                out = m.mixes_collapse(x, pre_in, w, 1e-20, emit_rm=emit_rm)
                mix, h = out[0], host(out[1])[0, 0, :T]
                chk = [(f"mix{i}", host(mix[i]).reshape(T, -1), ref_mix[i]) for i in range(3)] + [
                    ("h", h, ref_h if not emit_rm else ref_h2)
                ]
                if emit_rm:
                    chk.append(("tok", host(out[2]).reshape(T, -1), ref_tok))
                for name, a, b in chk:
                    tot += 1
                    if not torch.equal(a, b):
                        bad += 1
                        print("DIFF T", T, "rm", emit_rm, name, (a - b).abs().max().item(), flush=True)
    print(f"PRE_TOTAL differing {bad} of {tot}", flush=True)
    assert bad == 0
