"""mHC mixes of a T=8 block (natural T=8, or the k=5 verify block of 6 rows padded to 8 by DSV41Layer.forward / DSV41MHC.mixes) are bit-identical to the T=4 program's rows
(= the rows of plain decode, which pads to 4). Also times the T=8 mixes in a trace with the T=4-group plan (default) vs the old 8-token group plan (DSV41_MHC_MIXES_T4=0)."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC as New

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
def test_mhc_t6_bitexact(mesh_device):
    md = mesh_device
    torch.manual_seed(1)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    fn, base, scale = torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5])
    new = New(md, fn, base, scale)
    bad = {}
    for t4 in ("1", "0"):
        os.environ["DSV41_MHC_MIXES_T4"] = t4
        nb = 0
        for trial in range(6):
            x = torch.randn(8, 1, 1, D) + 0.3 * torch.randn(8, 1, 4, D)
            x[..., :4] *= 300.0
            ref = [
                [host(o) for o in new.mixes(up(x[lo : lo + 4].clone()))] for lo in (0, 4)
            ]  # the T=4 program, 4 rows each
            want = [torch.cat([ref[0][i], ref[1][i]], 0).reshape(8, -1) for i in range(3)]
            n8 = [host(o).reshape(8, -1) for o in new.mixes(up(x))]  # natural T=8
            n6 = [host(o).reshape(6, -1) for o in new.mixes(up(x[:6].clone()))]  # 6 rows -> padded to 8 inside mixes
            # the hoisted pad of DSV41Layer.forward: rows 6, 7 hold other data (real mixes output of the drafter path), not zeros
            xh = x.clone()
            xh[6:] = torch.randn(2, 1, 4, D)
            h6 = [host(o).reshape(8, -1)[:6] for o in new.mixes(up(xh))]
            for i in range(3):
                nb += int(not torch.equal(want[i], n8[i]))
                nb += int(not torch.equal(want[i][:6], n6[i]))
                nb += int(not torch.equal(want[i][:6], h6[i]))
        bad[t4] = nb
        print(f"MIXES_T4={t4}: differing tensors {nb} of 54", flush=True)
        os.environ["DSV41_MHC_MIXES_T4"] = t4
        x = up(torch.randn(8, 1, 4, D))
        new.mixes(x)
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(40):
            outs = new.mixes(x)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        us = (time.perf_counter() - t0) / 5 / 40 * 1e6
        ttnn.release_trace(md, tid)
        print(f"MIXES_T4={t4}: T=8 mixes {us:.1f} us per call (traced)", flush=True)
    assert bad["1"] == 0
