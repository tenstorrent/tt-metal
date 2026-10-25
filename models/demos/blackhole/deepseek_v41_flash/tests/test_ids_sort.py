"""tt.indexer.canonical_ids: ascending order of the selected entry ids (sentinels last) for both id layouts ([R,1,1,k] and [1,1,R,k]); timing in a trace."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import canonical_ids


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
def test_ids_sort(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    ok = True
    for R in (1, 4, 6):
        ids = torch.stack([torch.randperm(524288)[:512] for _ in range(R)]).to(torch.int64)
        ids[0, 500:] = 0xFFFFFFFF  # sentinels (fewer valid entries)
        for shp in ((R, 1, 1, 512), (1, 1, R, 512)):
            x = ttnn.from_torch(
                ids.reshape(*shp).to(torch.int32),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            y = canonical_ids(x)
            h = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).reshape(R, 512).to(torch.int64) & 0xFFFFFFFF
            want = torch.sort(ids.reshape(R, 512), dim=-1).values
            eq = bool(torch.equal(h, want)) and tuple(y.shape) == shp
            ok &= eq
            print(f"R={R} shape {shp}: out shape {tuple(y.shape)} equal {eq}", flush=True)
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(20):
                canonical_ids(x)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(5):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            print(
                f"R={R} shape {shp}: canonical_ids {(time.perf_counter() - t0) / 5 / 20 * 1e6:.1f} us per call (traced)",
                flush=True,
            )
            ttnn.release_trace(md, tid)
    assert ok
