"""L02 gate (a): T-aware mHC ops vs the padded reference (DSV41_MHC_TAWARE=0 = the base tree's pad-to-4 / pad-to-8 wrappers) and vs the T=4 program, BIT-IDENTICAL.
Counts differing tensors (must be 0)."""
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
def test_mhc_taware(mesh_device):
    md = mesh_device
    torch.manual_seed(3)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    fn, base, scale = torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5])
    ma, mf = DSV41MHC(md, fn, base, scale), DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, base, scale)
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    bad, tot = {}, {}

    def cmp(name, a, b):
        tot[name] = tot.get(name, 0) + 1
        if a.shape != b.shape or not torch.equal(a, b):
            bad[name] = bad.get(name, 0) + 1
            if a.shape == b.shape:
                print("DIFF", name, "maxabs", (a - b).abs().max().item(), flush=True)
            else:
                print("DIFF", name, "shape", a.shape, b.shape, flush=True)

    def run(T, x, pre, y_bf, y_f, y2, post, comb, flag):
        os.environ["DSV41_MHC_TAWARE"] = flag
        out = {}
        out["mixes"] = [host(o).reshape(T, -1) for o in ma.mixes(x)]
        h, htok = mf.collapse_norm_rm(x, pre, w, 1e-20)
        out["cn_h"] = host(h)[0, 0, :T]
        out["cn_tok"] = host(htok).reshape(T, -1)
        out["cn"] = host(mf.collapse_norm(x, pre, w, 1e-20))[0, 0, :T]
        out["exp_ybf_y2"] = host(ma.expand(y_bf, x, post, comb, y2)).reshape(T, 4, -1)[:, :4]
        out["exp_ybf"] = host(ma.expand(y_bf, x, post, comb, None)).reshape(T, 4, -1)[:, :4]
        out["exp_yf_tok"] = host(ma.expand(y_f, x, post, comb, None)).reshape(T, 4, -1)[:, :4]
        return out

    for trial in range(3):
        for T in (1, 2, 3, 4, 5, 6, 7, 8):
            x_h = torch.randn(T, 1, 1, D) + 0.3 * torch.randn(T, 1, 4, D)
            x_h[..., :4] *= 300.0
            x, pre = up(x_h), up(torch.rand(T, 1, 1, 4))
            y_bf, y2 = up(torch.randn(1, 1, T, D), ttnn.bfloat16), up(torch.randn(1, 1, T, D))
            y_f = up(torch.randn(T, 1, 1, D))
            post, comb = up(torch.rand(T, 1, 4, 1) * 2), up(torch.rand(T, 1, 4, 4))
            new = run(T, x, pre, y_bf, y_f, y2, post, comb, "1")
            old = run(T, x, pre, y_bf, y_f, y2, post, comb, "0")
            for k in new:
                if k == "mixes":
                    for i in range(3):
                        cmp(f"T{T}.mixes{i}.vs_old", new[k][i], old[k][i])
                else:
                    cmp(f"T{T}.{k}.vs_old", new[k], old[k])
            # vs the T=4 programs (rows 0..3 / 4..7 of a natural T=8 reference); T=4 vs T=4-chunks
            if T > 4:
                os.environ["DSV41_MHC_TAWARE"] = "1"
                lo, hi = x_h[:4].clone(), x_h[4:].clone()
                r_lo = [host(o).reshape(4, -1) for o in ma.mixes(up(lo))]
                hi4 = torch.zeros(4, 1, 4, D)
                hi4[: T - 4] = hi.reshape(T - 4, 1, 4, D) if hi.shape[2] == 4 else hi.expand(T - 4, 1, 4, D)
                x4b = up(hi4)
                r_hi = [host(o).reshape(4, -1) for o in ma.mixes(x4b)]
                nn_ = [host(o).reshape(T, -1) for o in ma.mixes(x)]
                for i in range(3):
                    cmp(f"T{T}.mixes{i}.vs_T4prog", torch.cat([r_lo[i], r_hi[i][: T - 4]], 0), nn_[i])
    for k in sorted(tot):
        print(f"GATE {k}: {bad.get(k, 0)}/{tot[k]} differing")
    print("TOTAL differing tensors:", sum(bad.values()), "of", sum(tot.values()), flush=True)
    assert sum(bad.values()) == 0
