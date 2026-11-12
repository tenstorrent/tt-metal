"""L1 footprint probe of ttnn.point_to_point program-cache entries (each cache entry parks a GlobalSemaphore: a persistent L1 allocation on every worker core)."""
import os

import pytest

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
    print("L1PROBE l1_small", L1S, "baseline", l1(md), flush=True)
    for name, layout, shape in (
        ("rm_lat", ttnn.ROW_MAJOR_LAYOUT, [1, 1, 2048, 512]),
        ("tile_key", ttnn.TILE_LAYOUT, [8, 1, 256, 128]),
        ("tile_key2", ttnn.TILE_LAYOUT, [8, 1, 512, 128]),
        ("tile_halo", ttnn.TILE_LAYOUT, [8, 1, 128, 512]),
    ):
        d = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=layout, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=layout, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        n = 0
        for sr in range(4):
            for dr in range(4):
                if sr == dr:
                    continue
                for c in range(8):
                    ttnn.point_to_point(
                        d,
                        ttnn.MeshCoordinate(sr, c),
                        ttnn.MeshCoordinate(dr, c),
                        topology=ttnn.Topology.Linear,
                        output_tensor=o,
                    )
                    n += 1
        print("L1PROBE", name, "entries", n, l1(md), flush=True)
        # second round: program cache hits must not allocate
    print("L1PROBE done", flush=True)
