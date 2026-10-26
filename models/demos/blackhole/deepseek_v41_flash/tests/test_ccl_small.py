"""L07: tiny all-gathers (sampler / drafter / Engram candidates): link / worker / buffer variants, traced us + exactness."""
import itertools

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 300_000_000}],
    indirect=True,
)
@torch.no_grad()
def test_ccl_small(mesh_device):
    md = mesh_device
    rows, cols = 4, 8
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    own = [[ttnn.create_global_semaphore(md, ccl.ccl_cores, 0) for _ in range(2)] for _ in range(2)]
    bar = [ttnn.create_global_semaphore(md, ccl.ccl_cores, 0) for _ in range(2)]
    for R, n, dt in ((4, 1, ttnn.float32), (16, 1, ttnn.float32), (16, 64, ttnn.bfloat16), (16, 640, ttnn.bfloat16)):
        g = torch.Generator().manual_seed(n)
        h = torch.randn(rows, cols, R, n, generator=g)
        x = ttnn.from_torch(
            h,
            device=md,
            layout=ttnn.TILE_LAYOUT,
            dtype=dt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
        )
        ref = h.permute(0, 2, 1, 3).reshape(rows, R, cols * n)

        def mk(links, workers, bufs, mine):
            ctr = [0]

            def f():
                ctr[0] ^= 1
                return ttnn.experimental.all_gather_async(
                    x,
                    dim=3,
                    cluster_axis=1,
                    mesh_device=md,
                    topology=ttnn.Topology.Ring,
                    multi_device_global_semaphore=own[0][ctr[0]]
                    if False
                    else (own[ctr[0]] if mine else ccl.get_ag_ping_pong_semaphore()),
                    num_links=links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=bar[ctr[0]] if mine else ccl.get_barrier_semaphore(),
                    num_workers_per_link=workers,
                    num_buffers_per_channel=bufs,
                )

            return f

        for links, workers, bufs, mine in [(2, None, None, False), (2, None, None, True)] + list(
            itertools.product((1, 2), (1, 2), (None,), (True,))
        ):
            f = mk(links, workers, bufs, mine)
            f()
            ttnn.synchronize_device(md)
            y = f()
            ttnn.synchronize_device(md)
            c = ttnn.to_torch(
                y, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(rows, cols))
            ).float()[:, 0]
            err = float((c.reshape(rows, R, -1) - ref.to(torch.float32)).abs().max())
            print(
                f"SMALL R={R} n={n} {dt} links={links} workers={workers} own_sems={mine}: {chain_ms(md, f) * 1e3:.1f} us maxerr {err:.2e}",
                flush=True,
            )
