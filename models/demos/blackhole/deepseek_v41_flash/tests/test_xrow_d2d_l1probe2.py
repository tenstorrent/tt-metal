"""L1 footprint + correctness + time of ttnn.all_gather over the ROW axis (cluster_axis=0) as a row-to-row copy primitive (alternative to per-pair p2p program-cache entries)."""
import os
import time

import pytest
import torch

import ttnn

L1S = int(os.environ.get("D2D_L1_SMALL", "18432"))


def l1(md):
    ttnn.synchronize_device(md)
    mv = ttnn.get_memory_view(md, ttnn.BufferType.L1)
    return mv.total_bytes_allocated_per_bank, mv.largest_contiguous_bytes_free_per_bank


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": L1S, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
def test_probe(mesh_device):
    md = mesh_device
    print("L1PROBE baseline", l1(md), flush=True)
    for name, layout, shape in (
        ("rm_lat", ttnn.ROW_MAJOR_LAYOUT, [1, 1, 2048, 512]),
        ("rm_lat_big", ttnn.ROW_MAJOR_LAYOUT, [1, 1, 16384, 512]),
        ("tile_key", ttnn.TILE_LAYOUT, [128, 1, 256, 128]),
        ("tile_key2", ttnn.TILE_LAYOUT, [128, 1, 512, 128]),
        ("tile_halo", ttnn.TILE_LAYOUT, [32, 1, 128, 512]),
    ):
        A = shape[0]
        g = torch.Generator().manual_seed(1)
        host = torch.randn(4 * A, 8, *shape[2:], generator=g).bfloat16()  # [rows*A, cols, R, D]
        d = ttnn.from_torch(
            host,
            device=md,
            dtype=ttnn.bfloat16,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(4, 8)),
        )
        ts = []
        for it in range(3):
            t0 = time.perf_counter()
            o = ttnn.all_gather(d, dim=0, cluster_axis=0, num_links=1, topology=ttnn.Topology.Ring)
            ttnn.synchronize_device(md)
            ts.append(round(time.perf_counter() - t0, 4))
            if it < 2:
                ttnn.deallocate(o)
        h = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(4, 8)))
        ok = all(torch.equal(h[r * 4 * A : (r + 1) * 4 * A], host) for r in range(4))
        print("L1PROBE", name, "times", ts, "out_shape", tuple(o.shape), "equal_all_rows", ok, l1(md), flush=True)
        ttnn.deallocate(o)
        ttnn.deallocate(d)
    print("L1PROBE done", flush=True)
