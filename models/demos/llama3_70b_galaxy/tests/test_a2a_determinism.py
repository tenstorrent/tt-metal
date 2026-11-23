# Is all_to_all_async_generic (column axis, the WO-a2a layout) deterministic under back-to-back use? Runs a2a + full WO
# matmul N times on the same input, interleaved with other column CCLs like in the layer (all-gather), and compares
# every output with the first one, bit for bit, on several chips. Also MMRS as the control.
#   pytest models/demos/llama3_70b_galaxy/tests/test_a2a_determinism.py -s     env: AD_M (1024,2048,4096), AD_N (60)
import os

import pytest
import torch
from loguru import logger

import ttnn

H = 8192
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_a2a_determinism(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
    bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    lofi = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    mmrs_ck = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi)
    heads = ttnn.ShardTensor2dMesh(mesh, dims=(3, None), mesh_shape=(8, 4))
    rows = ttnn.ShardTensor2dMesh(mesh, dims=(0, None), mesh_shape=(8, 4))
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    wo = torch.randn(H, H) * 0.02
    t_wo_full = ttnn.from_torch(
        wo.reshape(1, 1, H, H), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rep
    )
    t_wo_sh = ttnn.from_torch(wo, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rows)
    n = int(os.environ.get("AD_N") or 60)
    chips = (0, 5, 13, 22, 31)

    def mm(a, w):
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=16,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=lofi,
            memory_config=DRAM,
            dtype=ttnn.bfloat8_b,
        )

    agi = [0]

    def ag(x):
        i = agi[0]
        agi[0] ^= 1
        return ttnn.experimental.all_gather_async(
            input_tensor=x,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=sems[i],
            num_links=2,
            barrier_semaphore=bars[i],
            memory_config=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            num_workers_per_link=8,
        )

    def snap(t):
        d = ttnn.get_device_tensors(t)
        return [ttnn.to_torch(d[c]).float() for c in chips]

    for M in [int(m) for m in (os.environ.get("AD_M") or "1024,2048,4096").split(",")]:
        att = torch.randn(1, 1, M, H)
        ta = ttnn.from_torch(att, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=heads)
        xg = ttnn.from_torch(
            torch.randn(1, 1, M // 8, H), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rep
        )
        for name, fn in (
            (
                "a2a+wo",
                lambda: mm(
                    ttnn.experimental.all_to_all_async_generic(
                        ta, 3, 2, num_links=2, memory_config=DRAM, topology=ttnn.Topology.Ring, cluster_axis=0
                    ),
                    t_wo_full,
                ),
            ),
            (
                "a2a",
                lambda: ttnn.experimental.all_to_all_async_generic(
                    ta, 3, 2, num_links=2, memory_config=DRAM, topology=ttnn.Topology.Ring, cluster_axis=0
                ),
            ),
            (
                "mmrs",
                lambda: ttnn.experimental.matmul_reduce_scatter_prefill(
                    ta,
                    t_wo_sh,
                    cluster_axis=0,
                    scatter_dim=-2,
                    topology=ttnn.Topology.Ring,
                    num_links=2,
                    compute_kernel_config=mmrs_ck,
                    memory_config=DRAM,
                ),
            ),
        ):
            ref = fn()
            r0 = snap(ref)
            ttnn.deallocate(ref)
            bad, worst = 0, 0.0
            for it in range(n):
                g = ag(xg)  # another column CCL in between, as in the layer
                o = fn()
                ttnn.deallocate(g)
                s = snap(o)
                ttnn.deallocate(o)
                d = max((a - b).abs().max().item() for a, b in zip(r0, s))
                if d > 0:
                    bad += 1
                    worst = max(worst, d)
            logger.info(f"ADET M={M} {name}: {bad}/{n} runs differ from the first (worst max abs diff {worst:.4g})")
            # unsynced: enqueue all runs back to back (each followed by a consumer, like the residual add), read at end
            outs = []
            for it in range(n):
                g = ag(xg)
                o = fn()
                ttnn.deallocate(g)
                y = ttnn.add(o, 0.0, memory_config=DRAM)
                ttnn.deallocate(o)
                outs.append(y)
            bad, worst = 0, 0.0
            for y in outs:
                s = snap(y)
                ttnn.deallocate(y)
                d = max((a - b).abs().max().item() for a, b in zip(r0, s))
                if d > 0:
                    bad += 1
                    worst = max(worst, d)
            logger.info(f"ADET M={M} {name} unsynced: {bad}/{n} differ (worst {worst:.4g})")
        ttnn.deallocate(ta), ttnn.deallocate(xg)
