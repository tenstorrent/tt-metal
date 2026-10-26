"""L07 correctness probe of all_gather_async / reduce_scatter_minimal_async at small row counts (who is wrong at R=4/8?)."""

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

D = 5120


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@torch.no_grad()
def test_ccl_corr(mesh_device):
    md = mesh_device
    rows, cols = 4, 8
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    own = [ttnn.create_global_semaphore(md, ccl.ccl_cores, 0) for _ in range(10)]
    for R in (4, 16):
        g = torch.Generator().manual_seed(R)
        sc_h = torch.randn(rows, cols, R, D // 8, generator=g).to(
            torch.bfloat16
        )  # the scattered result held by each device
        sc = ttnn.from_torch(
            sc_h,
            device=md,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
        )
        ref = sc_h.permute(0, 2, 1, 3).reshape(rows, R, D).float()  # gathered along dim 3 over columns

        def ag(workers, sems):
            return ttnn.experimental.all_gather_async(
                sc,
                dim=3,
                cluster_axis=1,
                mesh_device=md,
                topology=ccl.topology,
                multi_device_global_semaphore=sems(),
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=ccl.get_barrier_semaphore(),
                num_workers_per_link=workers,
            )

        def check(y):
            ttnn.synchronize_device(md)
            c = ttnn.to_torch(
                y, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(rows, cols))
            ).float()  # [rows, cols, R, D]
            bad = [int((c[:, j].reshape(rows, R, D) != ref).sum()) for j in range(cols)]
            return bad

        for name, w in (("default", None), ("w1", 1), ("w2", 2), ("default", None), ("w1", 1)):
            for rep in range(2):
                print(
                    f"CORR R={R} AG {name} rep{rep}: mismatching elements per column {check(ag(w, ccl.get_ag_ping_pong_semaphore))}",
                    flush=True,
                )
