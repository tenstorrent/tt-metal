"""device-profiler zone breakdown of the mHC ops (env DSV41_MHC_PROF_T). Run with TT_METAL_DEVICE_PROFILER=1; prints per-op zone spans (us) on device 0."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC

D = 5120


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
def test_mhc_prof(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    T = int(os.environ.get("DSV41_MHC_PROF_T", "4"))
    ma = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    wa = up(torch.rand(1, 1, 1, D) + 0.5)
    x = up(torch.randn(T, 1, 4, D))
    pre = up(torch.rand(T, 1, 1, 4))
    a = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
    post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))

    def once():
        ma.mixes_collapse(x, pre, wa, 1e-20, emit_rm=True)
        ma.expand(a, x, post, comb, None)

    once()
    ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    for _ in range(3):
        once()
        ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    print("PROF_DONE", flush=True)
